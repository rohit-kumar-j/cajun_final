"""Evaluate with frame metadata and live velocity graph overlay on video."""
from absl import app, flags
from datetime import datetime
import os, signal, sys, time, subprocess, shutil
from isaacgym import gymapi
from isaacgym.torch_utils import to_torch
from rsl_rl.runners import OnPolicyRunner
import torch, yaml, numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

_exit_state = {'data_arrays': None, 'data_count': 0, 'output_dir': None, 'steps_count': 0,
               'start_time': None, 'data_saved': False, 'gait_name': 'unknown'}

# Frame capture globals
_frame_count = 0
_physics_step_count = 0
_frame_skip = 1
_frame_template = ""
_frames_dir = ""
_metadata_file = None  # CSV file for frame metadata
_gym = None
_viewer = None
_env = None

# Current state for metadata (updated from main loop)
_current_sim_time = 0.0
_current_vel = 0.0
_desired_vel = 0.0

def calculate_and_plot_cot(output_dir, data_arrays, data_count, gait_name):
    """Calculate COT per stride and save plot."""
    if data_count < 100:
        print("Not enough data for COT calculation")
        return
    
    # Extract data
    torso_x = data_arrays['torso_x'][:data_count]
    time_data = data_arrays['time'][:data_count]
    front_stance = data_arrays['front_stance'][:data_count]
    joint_torque = data_arrays['joint_torque'][:data_count]  # (N, 12)
    joint_vel = data_arrays['joint_vel'][:data_count]  # (N, 12)
    
    # Parameters (Go1 robot)
    mass = 12.5  # kg
    g = 9.81
    dt = 0.002
    
    # Detect stride boundaries using front stance transitions
    front_binary = (front_stance == 5).astype(int)
    front_diff = np.diff(front_binary)
    liftoffs = np.where(front_diff == -1)[0]
    
    if len(liftoffs) < 2:
        print("Not enough strides for COT calculation")
        return
    
    velocities = []
    cots = []
    
    for i in range(len(liftoffs) - 1):
        start_idx = liftoffs[i] + 1
        end_idx = liftoffs[i + 1]
        
        if end_idx <= start_idx:
            continue
        
        # Distance traveled
        distance = torso_x[end_idx] - torso_x[start_idx]
        if distance <= 0:
            continue
        
        # Stride time
        stride_time = time_data[end_idx] - time_data[start_idx]
        if stride_time <= 0:
            continue
        
        # Velocity
        velocity = distance / stride_time
        
        # Mechanical work (absolute value of power integrated)
        tau = joint_torque[start_idx:end_idx]  # (stride_len, 12)
        dq = joint_vel[start_idx:end_idx]  # (stride_len, 12)
        power = tau * dq
        work = np.sum(np.abs(power)) * dt
        
        # Gravitational work (normalization)
        grav_work = mass * g * abs(distance)
        
        # COT
        cot = work / grav_work
        
        # Filter reasonable values
        if 0 < velocity < 10 and 0 < cot < 20:
            velocities.append(velocity)
            cots.append(cot)
    
    if len(velocities) == 0:
        print("No valid strides for COT calculation")
        return
    
    velocities = np.array(velocities)
    cots = np.array(cots)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(velocities, cots, c='#1f77b4', marker='o', s=50, alpha=0.7)
    ax.set_xlabel('Velocity (m/s)', fontsize=12)
    ax.set_ylabel('Cost of Transport (COT)', fontsize=12)
    ax.set_title(f'Cost of Transport vs Velocity - {gait_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(velocities) * 1.1)
    ax.set_ylim(0, max(cots) * 1.1)
    
    fig.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'cot_plot.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ COT plot saved: {plot_path}")
    print(f"  Strides analyzed: {len(velocities)}")
    print(f"  Velocity range: {velocities.min():.2f} - {velocities.max():.2f} m/s")
    print(f"  COT range: {cots.min():.2f} - {cots.max():.2f}")

def draw_box(gym, viewer, env, center, color, thickness=3):
    """Draw a box on ground."""
    x, y, z, hl, hw = center if len(center) == 5 else (*center, 0.01)
    corners = [[x-hl, y-hw, z], [x+hl, y-hw, z], [x+hl, y+hw, z], [x-hl, y+hw, z]]
    lines = []
    for t in range(thickness):
        zo = t * 0.002
        for i in range(4):
            s, e = corners[i].copy(), corners[(i+1)%4].copy()
            s[2] += zo
            e[2] += zo
            lines.append(s + e)
    gym.add_lines(viewer, env._robot._envs[0], len(lines), 
                 np.array(lines, dtype=np.float32), 
                 np.array([color] * len(lines), dtype=np.float32))

def draw_overlay(sim_time):
    """Draw landing boxes - called before render."""
    global _physics_step_count
    _physics_step_count += 1
    
    # Only draw when capturing
    if (_physics_step_count - 1) % _frame_skip != 0:
        return
    
    if FLAGS.render_landing_pos and _env:
        _gym.clear_lines(_viewer)
        robot_pos = _env._robot.base_position[0].cpu().numpy()
        foot_pos = _env._swing_leg_controller.desired_foot_positions
        colors = [[1,0.2,0.2], [0.2,1,0.2], [0.3,0.3,1], [1,1,0.2]]
        for fid in range(4):
            fp = foot_pos[0, fid].cpu().numpy()
            draw_box(_gym, _viewer, _env, 
                    (fp[0]+robot_pos[0], fp[1]+robot_pos[1], 0.02, 0.05, 0.05), 
                    colors[fid])

def capture_frame_with_metadata():
    """Capture frame and save metadata."""
    global _frame_count, _physics_step_count, _metadata_file
    global _current_sim_time, _current_vel, _desired_vel
    
    # Only capture every Nth physics step
    if (_physics_step_count - 1) % _frame_skip != 0:
        return
    
    # Capture frame
    frame_path = _frame_template.format(_frame_count)
    _gym.write_viewer_image_to_file(_viewer, frame_path)
    
    # Write metadata to CSV
    if _metadata_file is not None:
        _metadata_file.write(f"{_frame_count},{_current_sim_time:.6f},{_current_vel:.6f},{_desired_vel:.6f}\n")
    
    _frame_count += 1

def create_video_with_graph_overlay(frames_dir, metadata_path, video_path, fps):
    """Create video with live velocity graph overlay using matplotlib."""
    if _frame_count == 0:
        print("No frames captured, skipping video creation")
        return False
    
    print(f"\n✓ Creating video with velocity graph overlay...")
    
    # Use local frame template based on frames_dir parameter
    frame_template = os.path.join(frames_dir, "frame_{:06d}.png")
    
    # Check if frames actually exist
    first_frame = frame_template.format(0)
    if not os.path.exists(first_frame):
        print(f"No frame files found at {first_frame}, skipping video creation")
        return False
    
    # Load metadata
    metadata = np.loadtxt(metadata_path, delimiter=',', skiprows=1)
    frame_nums = metadata[:, 0].astype(int)
    sim_times = metadata[:, 1]
    curr_vels = metadata[:, 2]
    des_vels = metadata[:, 3]
    
    # Create output directory
    overlay_dir = os.path.join(os.path.dirname(frames_dir), "frames_with_overlay")
    os.makedirs(overlay_dir, exist_ok=True)
    
    graph_history_duration = 2.0  
    # --- SETUP MATPLOTLIB ONCE ---
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    line_des, = ax.plot([], [], 'r--', label='Desired', linewidth=2)
    line_curr, = ax.plot([], [], 'b-', label='Current', linewidth=2)
    
    ax.set_ylim(0, max(6.5, des_vels.max() * 1.1))
    ax.set_xlabel('Time (s)', fontsize=8)
    ax.set_ylabel('Velocity (m/s)', fontsize=8)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    frames_processed = 0
    # Process each frame
    for i, frame_num in enumerate(tqdm(frame_nums, desc="Adding overlays", unit="frame")):
        frame_path = frame_template.format(frame_num)  # Use local variable
        if not os.path.exists(frame_path):
            continue
        
        img = cv2.imread(frame_path)
        if img is None:
            continue
        
        frames_processed += 1
        current_time = sim_times[i]
        
        # Slicing data for history window
        history_mask = (sim_times <= current_time) & (sim_times >= current_time - graph_history_duration)
        h_times = sim_times[history_mask]
        h_curr = curr_vels[history_mask]
        h_des = des_vels[history_mask]
        
        # --- UPDATE PLOT DATA ---
        line_des.set_data(h_times, h_des)
        line_curr.set_data(h_times, h_curr)
        ax.set_xlim(current_time - graph_history_duration, current_time)
        ax.set_title(f't={current_time:.2f}s | v={curr_vels[i]:.2f}m/s', fontsize=9)
        
        # Convert plot to image buffer
        fig.canvas.draw()
        graph_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGB2BGR)
        
        # Overlay Logic
        graph_h, graph_w = 200, 400
        graph_resized = cv2.resize(graph_img, (graph_w, graph_h))
        
        margin = 10
        y1, y2 = margin, margin + graph_h
        x1, x2 = margin, margin + graph_w
        
        # Semi-transparent background for the graph area
        overlay = img.copy()
        cv2.rectangle(overlay, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        
        # Place graph onto frame
        img[y1:y2, x1:x2] = graph_resized
        
        output_path = os.path.join(overlay_dir, f"frame_{frame_num:06d}.png")
        cv2.imwrite(output_path, img)
    
    plt.close(fig)
    
    print(f"  Processed {frames_processed} frames")
    
    if frames_processed == 0:
        print("  No frames were processed, skipping video encoding")
        return False
    
    # --- ENCODE VIDEO ---
    print(f"  Encoding video...")
    try:
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-pattern_type', 'glob',
            '-i', os.path.join(overlay_dir, 'frame_*.png'),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '20',
            '-pix_fmt', 'yuv420p',
            video_path
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Cleanup
        shutil.rmtree(frames_dir)
        shutil.rmtree(overlay_dir)
        print(f"  ✓ Video saved: {video_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ FFmpeg Error (exit code {e.returncode}):")
        print(f"    stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def save_matlab_data(output_dir, data_arrays, data_count):
    if data_count == 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving {data_count} MATLAB samples...")
    
    def save_txt(name, data):
        np.savetxt(os.path.join(output_dir, name), data[:data_count], fmt='%.6f')
    
    save_txt('time.txt', data_arrays['time'])
    save_txt('PosTorso0.txt', data_arrays['torso_x'])
    save_txt('PosTorso1.txt', data_arrays['torso_pitch'])
    save_txt('PosTorso2.txt', data_arrays['torso_roll'])
    save_txt('PosTorso3.txt', data_arrays['torso_yaw'])
    save_txt('PosTorso4.txt', data_arrays['torso_y'])
    save_txt('PosTorso5.txt', data_arrays['torso_z'])
    save_txt('PosTorso6.txt', data_arrays['torso_vx'])
    save_txt('PosTorso7.txt', data_arrays['torso_vy'])
    save_txt('PosTorso8.txt', data_arrays['torso_vz'])
    
    for i in range(12):
        save_txt(f'q{i}.txt', data_arrays['joint_pos'][:, i])
        save_txt(f'dq{i}.txt', data_arrays['joint_vel'][:, i])
        save_txt(f'tauM{i}.txt', data_arrays['joint_torque'][:, i])
        save_txt(f'simforceFeetGlobal{i}.txt', data_arrays['foot_forces'][:, i])
    
    save_txt('desPosTorso9.txt', data_arrays['front_stance'])
    save_txt('desPosTorso10.txt', data_arrays['rear_stance'])
    save_txt('desired_vel_x.txt', data_arrays['desired_vel_x'])
    
    for i, name in enumerate(['FR', 'FL', 'RR', 'RL']):
        save_txt(f'contact_{name}.txt', data_arrays['contacts'][:, i])
    
    print(f"✓ MATLAB data saved to {output_dir}")

def save_on_exit(reason="unknown"):
    global _exit_state, _frame_count, _frames_dir, _metadata_file
    if _exit_state.get('data_saved'):
        return
    _exit_state['data_saved'] = True
    
    print(f"\n{'='*60}\nSaving ({reason})\n{'='*60}")
    print(f"Steps: {_exit_state['steps_count']}, Frames: {_frame_count}")
    
    # Close metadata file
    if _metadata_file:
        _metadata_file.close()
    
    # Create video with graph overlay
    if _frame_count > 0 and _frames_dir:
        metadata_path = os.path.join(os.path.dirname(_frames_dir), "frame_metadata.csv")
        create_video_with_graph_overlay(_frames_dir, metadata_path, 
                                        _exit_state.get('video_path'), 
                                        FLAGS.target_fps)
    
    # Save MATLAB data
    if _exit_state.get('data_arrays'):
        save_matlab_data(_exit_state['output_dir'], _exit_state['data_arrays'], 
                        _exit_state['data_count'])
        
        # Calculate and plot COT
        calculate_and_plot_cot(_exit_state['output_dir'], 
                               _exit_state['data_arrays'],
                               _exit_state['data_count'],
                               _exit_state['gait_name'])

signal.signal(signal.SIGINT, lambda s, f: (save_on_exit(f"signal {s}"), sys.exit(0)))

from src.envs import env_wrappers

flags.DEFINE_string("logdir", None, "logdir.")
flags.DEFINE_bool("use_gpu", False, "use GPU.")
flags.DEFINE_bool("show_gui", True, "show GUI.")
flags.DEFINE_bool("use_real_robot", False, "use real robot.")
flags.DEFINE_integer("num_envs", 1, "num environments.")
flags.DEFINE_integer("max_steps", 10000, "max steps.")
flags.DEFINE_bool("record_video", False, "record video.")
flags.DEFINE_bool("render_landing_pos", True, "render landing boxes.")
flags.DEFINE_integer("target_fps", 60, "Target FPS (30, 60, 120, 250).")
flags.DEFINE_bool("show_velocity_graph", True, "Show live velocity graph on video.")
FLAGS = flags.FLAGS

def get_latest_policy(logdir):
    files = sorted([f for f in os.listdir(logdir) if f.startswith("model")],
                  key=lambda e: os.path.getmtime(os.path.join(logdir, e)), reverse=True)
    return os.path.join(logdir, files[0]) if files else None

def main(argv):
    global _exit_state, _frame_count, _physics_step_count, _frame_skip
    global _frame_template, _frames_dir, _metadata_file
    global _gym, _viewer, _env
    global _current_sim_time, _current_vel, _desired_vel
    
    device = "cuda" if FLAGS.use_gpu else "cpu"
    
    if FLAGS.logdir.endswith("pt"):
        config_path = os.path.join(os.path.dirname(FLAGS.logdir), "config.yaml")
        policy_path, root_path = FLAGS.logdir, os.path.dirname(FLAGS.logdir)
    else:
        config_path = os.path.join(FLAGS.logdir, "config.yaml")
        policy_path, root_path = get_latest_policy(FLAGS.logdir), FLAGS.logdir
    
    config = yaml.load(open(config_path), Loader=yaml.Loader)
    gait_name = getattr(config.environment.gait, 'gait_name', 'unknown')
    _exit_state['gait_name'] = gait_name
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(root_path, f"detailed_{gait_name}_{timestamp}")
    _exit_state['output_dir'] = output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Gait: {gait_name}\nOutput: {output_dir}")
    
    with config.unlocked():
        velocity_schedule = torch.linspace(0.1, 5.0, 5000)
        config.environment.jumping_distance_schedule = velocity_schedule / config.environment.gait.stepping_frequency
        config.environment.max_jumps = 100000
    
    env = config.env_class(num_envs=FLAGS.num_envs, device=device, config=config.environment,
                          show_gui=FLAGS.show_gui or FLAGS.record_video, 
                          use_real_robot=FLAGS.use_real_robot)
    env = env_wrappers.RangeNormalize(env)
    _env = env._env  # Unwrapped env for direct access
    
    # Setup frame capture
    if FLAGS.record_video:
        _frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(_frames_dir, exist_ok=True)
        video_path = os.path.join(output_dir, f"{gait_name}_{FLAGS.target_fps}fps.mp4")
        metadata_path = os.path.join(output_dir, "frame_metadata.csv")
        
        _gym, _viewer = _env._gym, _env._viewer
        _frame_count, _physics_step_count = 0, 0
        _frame_template = os.path.join(_frames_dir, "frame_{:06d}.png")
        
        # Calculate frame skip (500 Hz → target fps)
        _frame_skip = max(1, int(500 / FLAGS.target_fps))
        actual_fps = 500 / _frame_skip
        
        # Open metadata CSV file
        _metadata_file = open(metadata_path, 'w')
        _metadata_file.write("frame,sim_time,current_vel,desired_vel\n")
        
        # Set callbacks
        _env._frame_callback = draw_overlay
        _env._capture_callback = capture_frame_with_metadata
        
        _exit_state['video_path'] = video_path
        
        print(f"✓ Frame capture:")
        print(f"  Physics: 500 Hz")
        print(f"  Capture: {actual_fps:.1f} fps (every {_frame_skip} physics steps)")
        print(f"  Graph overlay: {FLAGS.show_velocity_graph}")
        print(f"  Metadata: {metadata_path}")
    
    runner = OnPolicyRunner(env, config.training, policy_path, device=device)
    runner.load(policy_path)
    policy = runner.get_inference_policy()
    runner.alg.actor_critic.train()
    
    state, _ = env.reset()
    
    # Data arrays
    max_samples = FLAGS.max_steps + 1000
    _exit_state['data_arrays'] = {
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
    _exit_state['start_time'] = time.time()
    
    print(f"Starting simulation...")
    
    steps, data_idx, vel_idx = 0, 0, 0
    pbar = tqdm(total=FLAGS.max_steps, desc="Simulating", unit="step")
    
    try:
        with torch.inference_mode():
            while steps < FLAGS.max_steps:
                steps += 1
                action = policy(state)
                state, _, reward, done, info = env.step(action)

                if done.any() == True:
                    print("Env Reset")
                    vel_idx = 0
                
                # Extract data using unwrapped env
                t = _env._robot.time_since_reset.item()
                contacts = _env._robot.foot_contacts[0].cpu().numpy()
                base_pos = _env._robot.base_position[0].cpu().numpy()
                base_vel = _env._robot.base_velocity_world_frame[0].cpu().numpy()
                base_rpy = _env._robot.base_orientation_rpy[0].cpu().numpy()
                
                # Update globals for metadata
                curr_vel = np.linalg.norm(base_vel[:2])
                des_vel = velocity_schedule[vel_idx].item()
                _current_sim_time = t
                _current_vel = curr_vel
                _desired_vel = des_vel
                # print(f"des vel: {des_vel}")
                
                # Store data
                data_arrays = _exit_state['data_arrays']
                data_arrays['time'][data_idx] = t
                data_arrays['torso_x'][data_idx] = base_pos[0]
                data_arrays['torso_y'][data_idx] = base_pos[1]
                data_arrays['torso_z'][data_idx] = base_pos[2]
                data_arrays['torso_vx'][data_idx] = base_vel[0]
                data_arrays['torso_vy'][data_idx] = base_vel[1]
                data_arrays['torso_vz'][data_idx] = base_vel[2]
                data_arrays['torso_roll'][data_idx] = base_rpy[0]
                data_arrays['torso_pitch'][data_idx] = base_rpy[1]
                data_arrays['torso_yaw'][data_idx] = base_rpy[2]
                data_arrays['joint_pos'][data_idx] = _env._robot.motor_positions[0].cpu().numpy()
                data_arrays['joint_vel'][data_idx] = _env._robot.motor_velocities[0].cpu().numpy()
                data_arrays['joint_torque'][data_idx] = _env._robot.motor_torques[0].cpu().numpy()
                try:
                    data_arrays['foot_forces'][data_idx] = _env._robot.foot_contact_forces[0].cpu().numpy().flatten()
                except:
                    data_arrays['foot_forces'][data_idx] = np.zeros(12)
                data_arrays['contacts'][data_idx] = contacts
                data_arrays['front_stance'][data_idx] = 5.0 if (contacts[0] or contacts[1]) else 0.0
                data_arrays['rear_stance'][data_idx] = 5.0 if (contacts[2] or contacts[3]) else 0.0
                data_arrays['desired_vel_x'][data_idx] = des_vel
                
                # Progress
                real_time = time.time() - _exit_state['start_time']
                pbar.set_postfix({
                    'SimT': f'{t:.1f}s',
                    'Vel': f'{curr_vel:.2f}/{des_vel:.2f}',
                    'Frames': _frame_count
                })
                pbar.update(1)
                
                data_idx += 1
                _exit_state['data_count'], _exit_state['steps_count'] = data_idx, steps
                
                # Update velocity using unwrapped env
                if steps % 50 == 0 and vel_idx < len(velocity_schedule) - 1:
                    vel_idx += 1
                    _env._desired_velocity[:, 0] = velocity_schedule[vel_idx]
                    _env._desired_velocity[:, 1:] = 0
    
    except Exception as e:
        pbar.close()
        print(f"\nException: {e}")
        import traceback
        traceback.print_exc()
        save_on_exit("exception")
        raise
    
    pbar.close()
    save_on_exit("complete")

if __name__ == "__main__":
    app.run(main)

