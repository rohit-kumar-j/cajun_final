"""Evaluate a trained policy and save detailed data for MATLAB analysis."""
from absl import app
from absl import flags

from datetime import datetime
import os
import signal
import sys
import time
import shutil
import subprocess

from isaacgym import gymapi
from isaacgym.torch_utils import to_torch
from rsl_rl.runners import OnPolicyRunner
import torch
import yaml
import numpy as np
from tqdm import tqdm


# Global state for saving data on exit
_exit_state = {
    'data_arrays': None,
    'data_count': 0,
    'output_dir': None,
    'steps_count': 0,
    'start_time': None,
    'data_saved': False,
    'gait_name': 'unknown',
    'frame_count': 0,
}

# Frame capture globals
_frame_count = 0
_next_frame_time = 0.0
_frame_interval = 0.0
_frame_template = ""
_gym = None
_viewer = None
_env = None  # Store env reference for visualization


def draw_box(gym, viewer, env, center=None, corners=None, color=[1, 0, 0], thickness=3):
    """Draw a box on the ground plane."""
    if center is not None:
        if len(center) == 5:
            x, y, z, half_length, half_width = center
        elif len(center) == 4:
            x, y, half_length, half_width = center
            z = 0.01
        else:
            raise ValueError("center must be (x, y, z, half_length, half_width) or (x, y, half_length, half_width)")
        corners = [
            [x - half_length, y - half_width, z],
            [x + half_length, y - half_width, z],
            [x + half_length, y + half_width, z],
            [x - half_length, y + half_width, z]
        ]
    elif corners is not None:
        corners = [list(c) + [0.01] * (3 - len(c)) for c in corners]
        if len(corners) != 4:
            raise ValueError("Must provide exactly 4 corner points")
    else:
        raise ValueError("Must provide either 'center' or 'corners'")
    
    # Create multiple offset lines for thickness
    lines = []
    offset_step = 0.002  # 2mm spacing between parallel lines
    
    for t in range(thickness):
        # Offset in the Z direction to create thickness
        z_offset = t * offset_step
        
        for i in range(4):
            start = corners[i].copy()
            end = corners[(i + 1) % 4].copy()
            
            # Add z offset for thickness
            start[2] += z_offset
            end[2] += z_offset
            
            lines.append(start + end)
    
    # Also draw filled cross-hatch pattern for better visibility
    # Diagonal lines across the rectangle
    for t in range(max(1, thickness // 2)):
        z_offset = t * offset_step
        # Diagonal 1
        diag1_start = corners[0].copy()
        diag1_end = corners[2].copy()
        diag1_start[2] += z_offset
        diag1_end[2] += z_offset
        lines.append(diag1_start + diag1_end)
        
        # Diagonal 2
        diag2_start = corners[1].copy()
        diag2_end = corners[3].copy()
        diag2_start[2] += z_offset
        diag2_end[2] += z_offset
        lines.append(diag2_start + diag2_end)
    
    # Convert to numpy arrays
    lines = np.array(lines, dtype=np.float32)
    colors_array = np.array([color] * len(lines), dtype=np.float32)
    
    gym.add_lines(viewer, env._robot._envs[0], lines.shape[0], lines, colors_array)


def capture_frame(sim_time):
    """Frame capture callback with optional landing position visualization."""
    global _frame_count, _next_frame_time
    
    if sim_time >= _next_frame_time:
        # Visualize landing positions if enabled
        if FLAGS.render_landing_pos and _env is not None:
            _gym.clear_lines(_viewer)
            
            # Get desired foot positions from the last step
            # We need to access the swing leg controller
            env_id = 0
            robot_pos = _env._robot.base_position[env_id].cpu().numpy()
            
            # Get desired foot positions from swing controller
            desired_foot_positions = _env._swing_leg_controller.desired_foot_positions
            
            # Colors for each foot
            colors = [
                [1, 0.2, 0.2],  # FR: Bright Red
                [0.2, 1, 0.2],  # FL: Bright Green
                [0.3, 0.3, 1],  # RR: Bright Blue
                [1, 1, 0.2]     # RL: Bright Yellow
            ]
            
            # Draw box for each foot's desired landing position
            for foot_id in range(4):
                foot_pos = desired_foot_positions[env_id, foot_id].cpu().numpy()
                
                # Fixed width and length for visualization
                width = 0.05
                length = 0.05
                
                world_x = foot_pos[0] + robot_pos[0]
                world_y = foot_pos[1] + robot_pos[1]
                
                draw_box(_gym, _viewer, _env,
                        center=(world_x, world_y, 0.02, length, width),
                        color=colors[foot_id],
                        thickness=5)
        
        # Capture frame
        frame_path = _frame_template.format(_frame_count)
        _gym.write_viewer_image_to_file(_viewer, frame_path)
        _frame_count += 1
        _next_frame_time += _frame_interval


def save_data_to_matlab_format(output_dir, data_arrays, data_count):
    """Save all data arrays to individual .txt files in MATLAB-compatible format."""
    if data_count == 0:
        print("No data to save.")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving {data_count} samples to {output_dir}...")
    
    def save_txt(filename, data):
        filepath = os.path.join(output_dir, filename)
        np.savetxt(filepath, data[:data_count], fmt='%.6f')
    
    # Torso states
    save_txt('PosTorso0.txt', data_arrays['torso_x'])
    save_txt('PosTorso1.txt', data_arrays['torso_pitch'])
    save_txt('PosTorso2.txt', data_arrays['torso_roll'])
    save_txt('PosTorso3.txt', data_arrays['torso_yaw'])
    save_txt('PosTorso4.txt', data_arrays['torso_y'])
    save_txt('PosTorso5.txt', data_arrays['torso_z'])
    save_txt('PosTorso6.txt', data_arrays['torso_vx'])
    save_txt('PosTorso7.txt', data_arrays['torso_vy'])
    save_txt('PosTorso8.txt', data_arrays['torso_vz'])
    
    # Joint data: q, dq, tau (0-11)
    for i in range(12):
        save_txt(f'q{i}.txt', data_arrays['joint_pos'][:, i])
        save_txt(f'dq{i}.txt', data_arrays['joint_vel'][:, i])
        save_txt(f'tauM{i}.txt', data_arrays['joint_torque'][:, i])
        save_txt(f'simforceFeetGlobal{i}.txt', data_arrays['foot_forces'][:, i])
    
    # Contact states
    save_txt('desPosTorso9.txt', data_arrays['front_stance'])
    save_txt('desPosTorso10.txt', data_arrays['rear_stance'])
    save_txt('time.txt', data_arrays['time'])
    save_txt('desired_vel_x.txt', data_arrays['desired_vel_x'])
    
    # Individual contacts
    for i, name in enumerate(['FR', 'FL', 'RR', 'RL']):
        save_txt(f'contact_{name}.txt', data_arrays['contacts'][:, i])
    
    # Metadata
    with open(os.path.join(output_dir, 'metadata.txt'), 'w') as f:
        f.write(f"# Gait: {_exit_state['gait_name']}\n")
        f.write(f"# Samples: {data_count}\n")
        f.write(f"# dt: 0.002s (500 Hz)\n")
        f.write(f"# Duration: {data_count * 0.002:.2f}s\n")
    
    print(f"Saved to {output_dir}")
    return True


def create_video_from_frames(frames_dir, video_path, fps, frame_count):
    """Create video from saved frames using ffmpeg."""
    if frame_count == 0:
        print("No frames to create video from.")
        return False
    
    print(f"\nCreating video from {frame_count} frames...")
    try:
        cmd = [
            'ffmpeg', '-y', 
            '-framerate', str(fps),
            '-i', os.path.join(frames_dir, 'frame_%06d.png'),
            '-c:v', 'libx264', 
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            video_path
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Video saved: {video_path}")
        
        # Delete frames to save space
        shutil.rmtree(frames_dir)
        print(f"✓ Cleaned up temporary frames")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error creating video: {e.stderr}")
        print(f"  Frames preserved in: {frames_dir}")
        return False
    except FileNotFoundError:
        print(f"✗ Error: ffmpeg not found. Install with: sudo apt-get install ffmpeg")
        print(f"  Frames preserved in: {frames_dir}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        print(f"  Frames preserved in: {frames_dir}")
        return False


def save_data_on_exit(reason="unknown"):
    """Save all collected data on any exit condition."""
    global _exit_state, _frame_count
    
    if _exit_state['data_saved']:
        return
    _exit_state['data_saved'] = True
    
    print(f"\n{'='*60}")
    print(f"Saving data (reason: {reason})...")
    print(f"{'='*60}")
    
    elapsed = time.time() - _exit_state['start_time'] if _exit_state['start_time'] else 0
    print(f"Steps: {_exit_state['steps_count']}, Time: {elapsed:.2f}s, Samples: {_exit_state['data_count']}")
    
    # Create video if frames were captured
    if _frame_count > 0 and _exit_state['frames_dir']:
        create_video_from_frames(
            _exit_state['frames_dir'],
            _exit_state['video_path'],
            FLAGS.render_fps,
            _frame_count
        )
    
    # Save MATLAB data
    if _exit_state['data_count'] > 0 and _exit_state['data_arrays'] is not None:
        try:
            save_data_to_matlab_format(
                _exit_state['output_dir'],
                _exit_state['data_arrays'],
                _exit_state['data_count']
            )
        except Exception as e:
            print(f"ERROR: {e}")
            fallback = f"detailed_data_emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            save_data_to_matlab_format(fallback, _exit_state['data_arrays'], _exit_state['data_count'])


def signal_handler(signum, frame):
    """Handle Ctrl+C and other signals."""
    print(f"\n\nReceived signal {signum}!")
    save_data_on_exit(reason=f"signal {signum}")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


from src.envs import env_wrappers
torch.set_printoptions(precision=2, sci_mode=False)

flags.DEFINE_string("logdir", None, "logdir.")
flags.DEFINE_bool("use_gpu", False, "whether to use GPU.")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_bool("use_real_robot", False, "whether to use real robot.")
flags.DEFINE_integer("num_envs", 1, "number of environments to evaluate in parallel.")
flags.DEFINE_bool("use_contact_sensor", True, "whether to use contact sensor.")
flags.DEFINE_integer("max_steps", 25000, "maximum number of simulation steps.")
flags.DEFINE_bool("record_video", False, "whether to record video of simulation.")
flags.DEFINE_integer("render_fps", 60, "FPS for video recording.")
flags.DEFINE_bool("render_landing_pos", True, "whether to render landing position boxes.")

FLAGS = flags.FLAGS


def get_latest_policy_path(logdir):
    files = [e for e in os.listdir(logdir) if os.path.isfile(os.path.join(logdir, e))]
    files.sort(key=lambda e: os.path.getmtime(os.path.join(logdir, e)), reverse=True)
    for e in files:
        if e.startswith("model"):
            return os.path.join(logdir, e)
    raise ValueError("No Valid Policy Found.")


def main(argv):
    global _exit_state, _frame_count, _next_frame_time, _frame_interval, _frame_template, _gym, _viewer, _env
    del argv

    device = "cuda" if FLAGS.use_gpu else "cpu"

    # Load config and policy
    if FLAGS.logdir.endswith("pt"):
        config_path = os.path.join(os.path.dirname(FLAGS.logdir), "config.yaml")
        policy_path = FLAGS.logdir
        root_path = os.path.dirname(FLAGS.logdir)
    else:
        config_path = os.path.join(FLAGS.logdir, "config.yaml")
        policy_path = get_latest_policy_path(FLAGS.logdir)
        root_path = FLAGS.logdir

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # Get gait info
    gait_name = getattr(config.environment.gait, 'gait_name', 'unknown_gait')
    _exit_state['gait_name'] = gait_name
    
    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(root_path, f"detailed_{gait_name}_data_{timestamp}")
    _exit_state['output_dir'] = output_dir
    
    print(f"Gait: {gait_name}")
    print(f"Output: {output_dir}")

    with config.unlocked():
        velocity_up = torch.linspace(2.5, 6.0, 50)
        velocity_schedule = velocity_up
        config.environment.jumping_distance_schedule = velocity_schedule / config.environment.gait.stepping_frequency
        config.environment.gait.desired_velocity = torch.tensor([velocity_schedule[0].item(), 0, 0])
        config.environment.max_jumps = 100000

    # Always need GUI for rendering or recording
    show_gui_actual = FLAGS.show_gui or FLAGS.record_video
    env = config.env_class(
        num_envs=FLAGS.num_envs,
        device=device,
        config=config.environment,
        show_gui=show_gui_actual,
        use_real_robot=FLAGS.use_real_robot
    )

    env = env_wrappers.RangeNormalize(env)
    unwrapped_env = env._env
    _env = unwrapped_env  # Store for visualization in callback
    print(f"Initial stepping_frequency: {unwrapped_env._gait_generator.stepping_frequency}")
    
    if FLAGS.use_real_robot:
        env.robot.state_estimator.use_external_contact_estimator = (not FLAGS.use_contact_sensor)

    # Setup video recording with frame callback
    if FLAGS.record_video:
        os.makedirs(output_dir, exist_ok=True)
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        video_path = os.path.join(output_dir, f"detailed_{gait_name}_data_{timestamp}_{FLAGS.render_fps}fps.mp4")
        
        # Initialize frame capture globals
        _frame_count = 0
        _next_frame_time = 0.0
        _frame_interval = 1.0 / FLAGS.render_fps
        _frame_template = os.path.join(frames_dir, "frame_{:06d}.png")
        _gym = unwrapped_env._gym
        _viewer = unwrapped_env._viewer
        
        # Store in exit state
        _exit_state['frames_dir'] = frames_dir
        _exit_state['video_path'] = video_path
        
        # Set callback
        unwrapped_env._frame_callback = capture_frame
        
        print(f"Frame capture enabled: {FLAGS.render_fps} fps")
        print(f"Landing position visualization: {FLAGS.render_landing_pos}")
        print(f"Frames directory: {frames_dir}")
        print(f"Video path: {video_path}")

    # Load policy
    runner = OnPolicyRunner(env, config.training, policy_path, device=device)
    runner.load(policy_path)
    policy = runner.get_inference_policy()
    runner.alg.actor_critic.train()

    state, _ = env.reset()
    
    # Pre-allocate data arrays
    max_samples = FLAGS.max_steps + 1000
    data_arrays = {
        'time': np.zeros(max_samples, dtype=np.float32),
        'torso_x': np.zeros(max_samples, dtype=np.float32),
        'torso_y': np.zeros(max_samples, dtype=np.float32),
        'torso_z': np.zeros(max_samples, dtype=np.float32),
        'torso_vx': np.zeros(max_samples, dtype=np.float32),
        'torso_vy': np.zeros(max_samples, dtype=np.float32),
        'torso_vz': np.zeros(max_samples, dtype=np.float32),
        'torso_roll': np.zeros(max_samples, dtype=np.float32),
        'torso_pitch': np.zeros(max_samples, dtype=np.float32),
        'torso_yaw': np.zeros(max_samples, dtype=np.float32),
        'joint_pos': np.zeros((max_samples, 12), dtype=np.float32),
        'joint_vel': np.zeros((max_samples, 12), dtype=np.float32),
        'joint_torque': np.zeros((max_samples, 12), dtype=np.float32),
        'foot_forces': np.zeros((max_samples, 12), dtype=np.float32),
        'contacts': np.zeros((max_samples, 4), dtype=np.float32),
        'front_stance': np.zeros(max_samples, dtype=np.float32),
        'rear_stance': np.zeros(max_samples, dtype=np.float32),
        'desired_vel_x': np.zeros(max_samples, dtype=np.float32),
    }
    _exit_state['data_arrays'] = data_arrays
    _exit_state['start_time'] = time.time()

    print(f"Starting simulation (max {FLAGS.max_steps} steps)...")
    
    steps_count = 0
    data_count = 0
    velocity_index = 0
    
    # Use tqdm for clean progress display
    pbar = tqdm(total=FLAGS.max_steps, desc="Evaluating", unit="step")

    # Check the gait generator class
    gg = unwrapped_env._gait_generator
    # print(f"GG type: {type(gg)}")
    # print(f"GG __dict__: {gg.__dict__}")
    
    try:
        with torch.inference_mode():
            while steps_count < FLAGS.max_steps:
                steps_count += 1
                
                action = policy(state)
                # unwrapped_env._gait_generator._stepping_frequency[:] = 7.0
                state, _, reward, done, info = env.step(action)
                # Frames are captured automatically inside env.step() via callback

                # Extract data
                t = env.robot.time_since_reset.item()
                contacts = env.robot.foot_contacts[0].cpu().numpy()
                base_pos = env.robot.base_position[0].cpu().numpy()
                base_vel = env.robot.base_velocity_world_frame[0].cpu().numpy()
                base_rpy = env.robot.base_orientation_rpy[0].cpu().numpy()
                joint_pos = env.robot.motor_positions[0].cpu().numpy()
                joint_vel = env.robot.motor_velocities[0].cpu().numpy()
                joint_tau = env.robot.motor_torques[0].cpu().numpy()
                
                try:
                    foot_forces = env.robot.foot_contact_forces[0].cpu().numpy().flatten()
                except:
                    foot_forces = np.zeros(12, dtype=np.float32)
                
                # Store data
                data_arrays['time'][data_count] = t
                data_arrays['torso_x'][data_count] = base_pos[0]
                data_arrays['torso_y'][data_count] = base_pos[1]
                data_arrays['torso_z'][data_count] = base_pos[2]
                data_arrays['torso_vx'][data_count] = base_vel[0]
                data_arrays['torso_vy'][data_count] = base_vel[1]
                data_arrays['torso_vz'][data_count] = base_vel[2]
                data_arrays['torso_roll'][data_count] = base_rpy[0]
                data_arrays['torso_pitch'][data_count] = base_rpy[1]
                data_arrays['torso_yaw'][data_count] = base_rpy[2]
                data_arrays['joint_pos'][data_count] = joint_pos
                data_arrays['joint_vel'][data_count] = joint_vel
                data_arrays['joint_torque'][data_count] = joint_tau
                data_arrays['foot_forces'][data_count] = foot_forces
                data_arrays['contacts'][data_count] = contacts
                data_arrays['front_stance'][data_count] = 5.0 if (contacts[0] or contacts[1]) else 0.0
                data_arrays['rear_stance'][data_count] = 5.0 if (contacts[2] or contacts[3]) else 0.0
                data_arrays['desired_vel_x'][data_count] = velocity_schedule[velocity_index].item()
                
                # Update progress bar with stats
                curr_vel = np.linalg.norm(base_vel[:2])
                des_vel = velocity_schedule[velocity_index].item()
                real_time = time.time() - _exit_state['start_time']
                freq = unwrapped_env._gait_generator.stepping_frequency[0].cpu().numpy()
                
                pbar.set_postfix({
                    'SimTime': f'{t:.2f}s',
                    'RealTime': f'{real_time:.2f}s',
                    'Freq': f'{freq:.2f}Hz',
                    'CurrVel': f'{curr_vel:.3f}m/s',
                    'DesVel': f'{des_vel:.3f}m/s',
                    'Frames': _frame_count if FLAGS.record_video else 0
                })
                pbar.update(1)
                # Add this debug print:
                # print(f"Gait config: {unwrapped_env._config.gait}")
                # print(f"GG type: {type(gg)}")
                # print(f"GG __dict__: {gg.__dict__}")
                
                data_count += 1
                _exit_state['data_count'] = data_count
                _exit_state['steps_count'] = steps_count

                # Update velocity schedule
                if steps_count % 50 == 0 and velocity_index < len(velocity_schedule) - 1:
                    velocity_index += 1
                    env._env._desired_velocity[:, 0] = velocity_schedule[velocity_index]
                    env._env._desired_velocity[:, 1:] = 0

    except Exception as e:
        pbar.close()
        print(f"\n\nException: {e}")
        import traceback
        traceback.print_exc()
        save_data_on_exit(reason="exception")
        raise
    
    # Normal completion
    pbar.close()
    save_data_on_exit(reason="complete")


if __name__ == "__main__":
    app.run(main)
