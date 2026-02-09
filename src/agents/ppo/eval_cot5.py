"""Evaluate a trained policy and save detailed data for MATLAB analysis."""
from absl import app
from absl import flags

from datetime import datetime
import os
import cv
import signal
import sys
import time

from isaacgym.torch_utils import to_torch  # pylint: disable=unused-import
from rsl_rl.runners import OnPolicyRunner
import torch
import yaml

import numpy as np
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Global state for saving data on exit
_exit_state = {
    'data_arrays': None,
    'data_count': 0,
    'output_dir': None,
    'steps_count': 0,
    'start_time': None,
    'data_saved': False,
    'gait_name': 'unknown',
}


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


def save_data_on_exit(reason="unknown"):
    """Save all collected data on any exit condition."""
    global _exit_state
    
    if _exit_state['data_saved']:
        return
    _exit_state['data_saved'] = True
    
    print(f"\n{'='*60}")
    print(f"Saving data (reason: {reason})...")
    print(f"{'='*60}")
    
    elapsed = time.time() - _exit_state['start_time'] if _exit_state['start_time'] else 0
    print(f"Steps: {_exit_state['steps_count']}, Time: {elapsed:.2f}s, Samples: {_exit_state['data_count']}")
    
    if _exit_state['data_count'] > 0 and _exit_state['data_arrays'] is not None:
        try:
            save_data_to_matlab_format(_exit_state['output_dir'], _exit_state['data_arrays'], _exit_state['data_count'])
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


class FastContactPlotter:
    """Minimal real-time contact plotter with pre-allocated circular buffer."""
    
    def __init__(self, window_duration=0.5, dt=0.002):
        self.window_size = int(window_duration / dt)
        self.dt = dt
        
        # Pre-allocate circular buffers
        self.time_buffer = np.zeros(self.window_size, dtype=np.float32)
        self.contact_buffer = np.zeros((self.window_size, 4), dtype=np.bool_)
        self.idx = 0
        self.filled = False
        
        # Contact plot y-positions: [FR, FL, RR, RL] -> display [RR=0, FR=1, FL=2, RL=3]
        self.contact_y = np.array([1, 2, 0, 3], dtype=np.float32)
        
        # Setup figure
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.ax.set_ylim(-0.5, 3.5)
        self.ax.set_yticks([0, 1, 2, 3])
        self.ax.set_yticklabels(['RR', 'FR', 'FL', 'RL'])
        self.ax.set_xlabel('Time (s)')
        self.ax.set_title('Foot Contacts')
        self.ax.grid(True, alpha=0.3)
        
        # Pre-create scatter plots for each leg
        self.scatters = [self.ax.scatter([], [], c='#4BACC6', marker='s', s=60) for _ in range(4)]
        
        self.last_draw_time = 0
        self.draw_interval = 0.05  # 20 Hz max update rate
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def update(self, t, contacts):
        """Add contact data and update plot if needed."""
        # Store in circular buffer
        self.time_buffer[self.idx] = t
        self.contact_buffer[self.idx] = contacts
        self.idx = (self.idx + 1) % self.window_size
        if self.idx == 0:
            self.filled = True
        
        # Rate-limit drawing
        now = time.time()
        if now - self.last_draw_time < self.draw_interval:
            return
        self.last_draw_time = now
        
        # Get valid data range
        if self.filled:
            times = np.concatenate([self.time_buffer[self.idx:], self.time_buffer[:self.idx]])
            contacts_arr = np.concatenate([self.contact_buffer[self.idx:], self.contact_buffer[:self.idx]])
        else:
            times = self.time_buffer[:self.idx]
            contacts_arr = self.contact_buffer[:self.idx]
        
        if len(times) == 0:
            return
        
        # Update scatter plots
        for leg_idx in range(4):
            mask = contacts_arr[:, leg_idx]
            leg_times = times[mask]
            if len(leg_times) > 0:
                y_vals = np.full(len(leg_times), self.contact_y[leg_idx])
                self.scatters[leg_idx].set_offsets(np.c_[leg_times, y_vals])
            else:
                self.scatters[leg_idx].set_offsets(np.empty((0, 2)))
        
        # Update x-axis
        self.ax.set_xlim(times[0], times[-1] + 0.02)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self):
        plt.close(self.fig)


from src.envs import env_wrappers
torch.set_printoptions(precision=2, sci_mode=False)

flags.DEFINE_string("logdir", None, "logdir.")
flags.DEFINE_bool("use_gpu", False, "whether to use GPU.")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_bool("use_real_robot", False, "whether to use real robot.")
flags.DEFINE_integer("num_envs", 1, "number of environments to evaluate in parallel.")
flags.DEFINE_bool("use_contact_sensor", True, "whether to use contact sensor.")
flags.DEFINE_bool("enable_plotting", True, "whether to enable real-time plotting.")
flags.DEFINE_integer("max_steps", 10000, "maximum number of simulation steps.")
flags.DEFINE_bool("record_video", False, "whether to record video of simulation.")
flags.DEFINE_integer("render_fps", 30, "FPS for video recording.")

FLAGS = flags.FLAGS


def get_latest_policy_path(logdir):
    files = [e for e in os.listdir(logdir) if os.path.isfile(os.path.join(logdir, e))]
    files.sort(key=lambda e: os.path.getmtime(os.path.join(logdir, e)), reverse=True)
    for e in files:
        if e.startswith("model"):
            return os.path.join(logdir, e)
    raise ValueError("No Valid Policy Found.")


def main(argv):
    global _exit_state
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
    swing_ratio = list(config.environment.gait.swing_ratio)
    _exit_state['gait_name'] = gait_name
    
    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(root_path, f"detailed_{gait_name}_data_{timestamp}")
    _exit_state['output_dir'] = output_dir

    # Video recording setup
    video_writer = None
    if FLAGS.record_video:
        os.makedirs(output_dir, exist_ok=True)
        video_path = os.path.join(output_dir, f"detailed_{gait_name}_data_{timestamp}_{FLAGS.render_fps}fps.mp4")
        print(f"Video will be saved to: {video_path}")

    
    print(f"Gait: {gait_name}")
    print(f"Output: {output_dir}")

    with config.unlocked():
        velocity_up = torch.linspace(0.5, 6.0, 500)
        velocity_schedule = velocity_up
        config.environment.jumping_distance_schedule = velocity_schedule / config.environment.gait.stepping_frequency
        config.environment.gait.desired_velocity = torch.tensor([velocity_schedule[0].item(), 0, 0])
        config.environment.max_jumps = 100000

    show_gui_actual = FLAGS.show_gui and not FLAGS.record_video
    env = config.env_class(num_envs=FLAGS.num_envs, device=device, config=config.environment,
                           show_gui=show_gui_actual, use_real_robot=FLAGS.use_real_robot)
                           # show_gui=FLAGS.show_gui, use_real_robot=FLAGS.use_real_robot)
    env = env_wrappers.RangeNormalize(env)
    
    if FLAGS.use_real_robot:
        env.robot.state_estimator.use_external_contact_estimator = (not FLAGS.use_contact_sensor)

    runner = OnPolicyRunner(env, config.training, policy_path, device=device)
    runner.load(policy_path)
    policy = runner.get_inference_policy()
    runner.alg.actor_critic.train()

    state, _ = env.reset()
    
    # Pre-allocate ALL data arrays upfront (float32 for speed)
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
    
    # Initialize plotter
    plotter = FastContactPlotter(window_duration=0.5) if FLAGS.enable_plotting else None

    print(f"Starting simulation (max {FLAGS.max_steps} steps)...")
    
    steps_count = 0
    data_count = 0
    velocity_index = 0
    frame_interval = 1.0 / FLAGS.render_fps
    last_frame_time = 0
    
    try:
        with torch.inference_mode():
            while steps_count < FLAGS.max_steps:
                steps_count += 1
                
                action = policy(state)
                state, _, reward, done, info = env.step(action)

                # Extract data (minimize CPU transfers by batching)
                t = env.robot.time_since_reset.item()
                contacts = env.robot.foot_contacts[0].cpu().numpy()  # [FR, FL, RR, RL]
                base_pos = env.robot.base_position[0].cpu().numpy()
                base_vel = env.robot.base_velocity_world_frame[0].cpu().numpy()
                base_rpy = env.robot.base_orientation_rpy[0].cpu().numpy()
                joint_pos = env.robot.motor_positions[0].cpu().numpy()
                joint_vel = env.robot.motor_velocities[0].cpu().numpy()
                joint_tau = env.robot.motor_torques[0].cpu().numpy()

                # Video recording - capture frames at specified FPS
                if FLAGS.record_video and (t - last_frame_time) >= frame_interval:
                    # Get camera image from Isaac Gym
                    env._gym.render_all_camera_sensors(env._sim)
                    
                    # Get image from viewer camera
                    img = env._gym.get_camera_image(env._sim, env.envs[0], env._gym.get_viewer_camera_handle(env.viewer), gymapi.IMAGE_COLOR)
                    
                    # Reshape image (Isaac Gym returns flat array)
                    img = img.reshape(img.shape[0], -1, 4)  # RGBA format
                    img = img[:, :, :3]  # Convert to RGB
                    img = np.ascontiguousarray(img)
                    
                    # Initialize video writer on first frame
                    if video_writer is None:
                        height, width = img.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(video_path, fourcc, FLAGS.render_fps, (width, height))
                        print(f"Initialized video writer: {width}x{height} @ {FLAGS.render_fps}fps")
                    
                    # Write frame (convert RGB to BGR for OpenCV)
                    video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    last_frame_time = t

                
                # Foot forces
                try:
                    foot_forces = env.robot.foot_contact_forces[0].cpu().numpy().flatten()
                except:
                    foot_forces = np.zeros(12, dtype=np.float32)
                
                # Store data directly (no bounds check - preallocated)
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
                
                data_count += 1
                _exit_state['data_count'] = data_count
                _exit_state['steps_count'] = steps_count

                # Update plotter (rate-limited internally)
                if plotter is not None:
                    plotter.update(t, contacts)

                # Update velocity schedule
                if steps_count % 50 == 0 and velocity_index < len(velocity_schedule) - 1:
                    velocity_index += 1
                    env._desired_velocity[:, 0] = velocity_schedule[velocity_index]
                    env._desired_velocity[:, 1:] = 0

                # Progress every 2000 steps
                if steps_count % 2000 == 0:
                    print(f"Step {steps_count}/{FLAGS.max_steps}, t={t:.2f}s, vel={np.linalg.norm(base_vel):.2f}m/s")

    except Exception as e:
        print(f"\nException: {e}")
        import traceback
        traceback.print_exc()
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved (partial): {video_path}")
        save_data_on_exit(reason="exception")
        if plotter:
            plotter.close()
        raise
    # Normal completion
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved: {video_path}")
    
    # Normal completion
    save_data_on_exit(reason="complete")
    
    if plotter:
        print("Press Enter to close...")
        input()
        plotter.close()


if __name__ == "__main__":
    app.run(main)
