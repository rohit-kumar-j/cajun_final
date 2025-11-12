"""Evaluate a trained policy."""
from absl import app
from absl import flags

from datetime import datetime
import os
import pickle
import time

from isaacgym.torch_utils import to_torch  # pylint: disable=unused-import
from rsl_rl.runners import OnPolicyRunner
import torch
import yaml

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def calculate_cot_per_stride(joint_torques, joint_velocities, dt, x_start, x_end, time_start, time_end, robot_mass, g=9.81):
  """
    Calculate Cost of Transport and velocity for a single stride.

    Args:
        joint_torques: Array of shape (num_joints, num_timesteps) - torques during stride
        joint_velocities: Array of shape (num_joints, num_timesteps) - velocities during stride
        dt: Time step between samples (seconds)
        x_start: Torso X position at stride start (meters)
        x_end: Torso X position at stride end (meters)
        time_start: Time at stride start (seconds)
        time_end: Time at stride end (seconds)
        robot_mass: Total mass of the robot (kg)
        g: Gravitational acceleration (m/s², default 9.81)

    Returns:
        velocity: Average velocity during stride (m/s)
        cot: Cost of Transport (dimensionless)
    """
  # Calculate stride velocity
  distance = x_end - x_start
  time_diff = time_end - time_start
  velocity = distance / time_diff

  # Calculate mechanical work: sum of |torque * velocity * dt| over all joints and timesteps
  # mechanical_work = np.sum(np.abs(joint_torques * joint_velocities * dt))

  joint_work = np.sum(joint_torques * joint_velocities * dt, axis=1)  # Sum over time for each joint
  mechanical_work = np.sum(np.abs(joint_work))  # Then take absolute value and sum joints


  # Calculate gravitational work: mass * g * distance
  gravitational_work = robot_mass * g * np.abs(distance)

  # COT = mechanical work / gravitational work
  cot = mechanical_work / gravitational_work

  return velocity, cot

class RealtimePlotter:
  """Lightweight real-time plotter for simulation data with minimal overhead."""

  def __init__(self, max_points=1000, update_interval=0.05):
    """
      Args:
        max_points: Maximum number of points to display
        update_interval: Update plot every N seconds (reduces overhead)
    """
    self.max_points = max_points
    self.update_interval = update_interval
    self.last_update_time = 0  # Track last update time

    # Create figure with subplots
    plt.ion()  # Interactive mode
    self.fig, self.axes = plt.subplots(2, 1, figsize=(12, 8))
    self.fig.tight_layout(pad=3.0)

    # Initialize empty data containers
    self.velocity_data = {'x': deque(maxlen=max_points), 'y': deque(maxlen=max_points)}
    self.stride_data = {'x': deque(maxlen=max_points), 'y': deque(maxlen=max_points)}
    self.stride_markers = {'x': [], 'y': []}

    # COT vs Velocity data
    self.cot_velocity_data = {'x': deque(maxlen=max_points), 'y': deque(maxlen=max_points)}

    # Setup plots
    self._setup_combined_plot()
    self._setup_cot_plot()

    # Draw initial canvas
    self.fig.canvas.draw()
    self.fig.canvas.flush_events()

  def _setup_combined_plot(self):
    """Setup combined velocity and stride length plot with dual y-axes."""
    ax1 = self.axes[0]
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity (m/s)', color='b')
    ax1.set_title('Velocity and Stride Length vs Time (with Stride Events)')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='b')

    # Velocity line on primary y-axis
    self.velocity_line, = ax1.plot([], [], 'b-', linewidth=2, label='Velocity')

    # Create secondary y-axis for stride length
    self.ax2 = ax1.twinx()
    self.ax2.set_ylabel('Stride Length (m)', color='g')
    self.ax2.tick_params(axis='y', labelcolor='g')

    # Stride length line on secondary y-axis
    self.stride_line, = self.ax2.plot([], [], 'g-', linewidth=2, label='Stride Length')

    # Vertical lines for stride events
    self.stride_lines = []

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = self.ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 3)
    self.ax2.set_ylim(0, 2)

  def _setup_cot_plot(self):
    """Setup COT vs Average Velocity scatter plot."""
    ax = self.axes[1]
    ax.set_xlabel('Average Velocity (m/s)')
    ax.set_ylabel('Cost of Transport (COT)')
    ax.set_title('Cost of Transport vs Average Velocity')
    ax.grid(True, alpha=0.3)

    # Scatter plot for COT vs velocity
    self.cot_scatter = ax.scatter([], [], c='purple', marker='o', s=50, alpha=0.6, label='COT')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 5)

  def add_data(self, time_val, velocity=None, stride_length=None, is_stride_event=False):
    """
      Add data point(s) to the plots.

      Args:
          time_val: Current simulation time
          velocity: Base velocity (optional)
          stride_length: Current stride length (optional)
          is_stride_event: Whether a stride event occurred
    """
    # Add velocity data
    if velocity is not None:
      self.velocity_data['x'].append(time_val)
      self.velocity_data['y'].append(velocity)

    # Add stride length data
    if stride_length is not None:
      self.stride_data['x'].append(time_val)
      self.stride_data['y'].append(stride_length)

    # Add stride event marker
    if is_stride_event:
      self.stride_markers['x'].append(time_val)
      self.stride_markers['y'].append(0)

    # Update plot only at specified time intervals
    current_real_time = time.time()
    if (current_real_time - self.last_update_time) >= self.update_interval:
      self._update_plot()
      self.last_update_time = current_real_time

  def add_cot_data(self, velocity, cot):
    """
    Add COT vs velocity data point.

    Args:
        velocity: Average velocity during stride (m/s)
        cot: Cost of Transport (dimensionless)
    """
    self.cot_velocity_data['x'].append(velocity)
    self.cot_velocity_data['y'].append(cot)

  def _update_plot(self):
    """Update the plots efficiently."""
    try:
      ax1 = self.axes[0]

      # Update velocity plot
      if len(self.velocity_data['x']) > 0:
        self.velocity_line.set_data(
          list(self.velocity_data['x']), 
          list(self.velocity_data['y'])
        )
        self._auto_scale_axis(ax1, self.velocity_data)

      # Update stride plot
      if len(self.stride_data['x']) > 0:
        self.stride_line.set_data(
          list(self.stride_data['x']), 
          list(self.stride_data['y'])
        )
        self._auto_scale_axis_secondary(self.ax2, self.stride_data)

      # Update stride event markers (vertical lines)
      self._update_stride_markers()

      # Update COT scatter plot
      if len(self.cot_velocity_data['x']) > 0:
        self.cot_scatter.set_offsets(
          np.c_[list(self.cot_velocity_data['x']), list(self.cot_velocity_data['y'])]
        )
        self._auto_scale_cot_plot(self.axes[1], self.cot_velocity_data)

      # Redraw canvas
      self.fig.canvas.draw()
      self.fig.canvas.flush_events()
    except Exception as e:
      print(f"Plot update error: {e}")

  def _update_stride_markers(self):
    """Update vertical lines for stride events on combined plot."""
    if len(self.stride_markers['x']) > 0:
      # Remove old lines
      for line in self.stride_lines:
        line.remove()

      self.stride_lines.clear()

      # Add new lines for each stride event
      for stride_time in self.stride_markers['x']:
        line = self.axes[0].axvline(x=stride_time, color='red', linestyle='--', 
                                    linewidth=1.5, alpha=0.7, 
                                    label='Stride Event' if not self.stride_lines else '')
        self.stride_lines.append(line)

      # Update legend if we have stride events
      if self.stride_lines:
        lines1, labels1 = self.axes[0].get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()
        self.axes[0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

  def _auto_scale_axis(self, ax, data):
    """Auto-scale primary axis based on data."""
    if len(data['x']) > 0:
      x_data = list(data['x'])
      y_data = list(data['y'])

      # X-axis: show last 10 seconds
      max_time = max(x_data)
      ax.set_xlim(max(0, max_time - 10), max_time + 1)

      # Y-axis: add 10% margin
      if y_data:
        y_min, y_max = min(y_data), max(y_data)
        margin = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 0.1
        ax.set_ylim(y_min - margin, y_max + margin)

  def _auto_scale_axis_secondary(self, ax, data):
    """Auto-scale secondary axis based on data."""
    if len(data['x']) > 0:
      y_data = list(data['y'])

      # Y-axis: add 10% margin
      if y_data:
        y_min, y_max = min(y_data), max(y_data)
        margin = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 0.1
        ax.set_ylim(y_min - margin, y_max + margin)

  def _auto_scale_cot_plot(self, ax, data):
    """Auto-scale COT plot based on data."""
    if len(data['x']) > 0 and len(data['y']) > 0:
      x_data = list(data['x'])
      y_data = list(data['y'])

      # X-axis (velocity): add 10% margin
      x_min, x_max = min(x_data), max(x_data)
      x_margin = (x_max - x_min) * 0.1 if (x_max - x_min) > 0 else 0.1
      ax.set_xlim(max(0, x_min - x_margin), x_max + x_margin)

      # Y-axis (COT): add 10% margin
      y_min, y_max = min(y_data), max(y_data)
      y_margin = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 0.1
      ax.set_ylim(max(0, y_min - y_margin), y_max + y_margin)

  def close(self):
    """Close the plot."""
    plt.close(self.fig)

from src.envs import env_wrappers
torch.set_printoptions(precision=2, sci_mode=False)

flags.DEFINE_string("logdir", None, "logdir.")
flags.DEFINE_bool("use_gpu", False, "whether to use GPU.")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_bool("use_real_robot", False, "whether to use real robot.")
flags.DEFINE_integer("num_envs", 1,
                     "number of environments to evaluate in parallel.")
flags.DEFINE_bool("save_traj", False, "whether to save trajectory.")
flags.DEFINE_bool("use_contact_sensor", True, "whether to use contact sensor.")
flags.DEFINE_bool("enable_plotting", True, "whether to enable real-time plotting.")
flags.DEFINE_float("plot_update_interval", 0.05, "plot update interval in seconds.")
FLAGS = flags.FLAGS


def get_latest_policy_path(logdir):
  files = [
    entry for entry in os.listdir(logdir)
    if os.path.isfile(os.path.join(logdir, entry))
  ]
  files.sort(key=lambda entry: os.path.getmtime(os.path.join(logdir, entry)))
  files = files[::-1]

  for entry in files:
    if entry.startswith("model"):
      return os.path.join(logdir, entry)
  raise ValueError("No Valid Policy Found.")


def main(argv):
  del argv  # unused

  device = "cuda" if FLAGS.use_gpu else "cpu"

  # Load config and policy
  if FLAGS.logdir.endswith("pt"):
    config_path = os.path.join(os.path.dirname(FLAGS.logdir), "config.yaml")
    policy_path = FLAGS.logdir
    root_path = os.path.dirname(FLAGS.logdir)
  else:
    # Find the latest policy ckpt
    config_path = os.path.join(FLAGS.logdir, "config.yaml")
    policy_path = get_latest_policy_path(FLAGS.logdir)
    root_path = FLAGS.logdir

  with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.Loader)

  with config.unlocked():
    config.environment.jumping_distance_schedule = torch.linspace(1.0, 2.0, 100)
    config.environment.max_jumps = 300

  env = config.env_class(num_envs=FLAGS.num_envs,
                         device=device,
                         config=config.environment,
                         show_gui=FLAGS.show_gui,
                         use_real_robot=FLAGS.use_real_robot)
  env = env_wrappers.RangeNormalize(env)
  if FLAGS.use_real_robot:
    env.robot.state_estimator.use_external_contact_estimator = (
      not FLAGS.use_contact_sensor)

  # Retrieve policy
  runner = OnPolicyRunner(env, config.training, policy_path, device=device)
  runner.load(policy_path)
  policy = runner.get_inference_policy()
  runner.alg.actor_critic.train()

  # Reset environment
  state, _ = env.reset()
  total_reward = torch.zeros(FLAGS.num_envs, device=device)
  steps_count = 0

  start_time = time.time()
  logs = []

  # Initialize plotter
  plotter = None
  max_strides = 1000
  cot_data_arrays = {
    'velocity': np.zeros(max_strides, dtype=np.float32),
    'cot': np.zeros(max_strides, dtype=np.float32),
    # 'time': np.zeros(max_strides, dtype=np.float32),
    # 'stride_length': np.zeros(max_strides, dtype=np.float32)
  }
  cot_data_count = 0

  if FLAGS.enable_plotting:
    plotter = RealtimePlotter(
      max_points=2000, 
      update_interval=FLAGS.plot_update_interval
    )
    print(f"Real-time plotting enabled (update every {FLAGS.plot_update_interval}s)")

  # Stride detection parameters
  foot_fall_pattern = [False, False, True, True]
  time_to_skip = 0.2  # Debounce time
  prev_skip_time = 0

  # Before the main loop - pre-allocate buffers
  max_stride_steps = 1000  # Adjust based on expected max stride duration
  num_joints = 12

  stride_torques = np.zeros((num_joints, max_stride_steps), dtype=np.float32)
  stride_velocities = np.zeros((num_joints, max_stride_steps), dtype=np.float32)
  stride_step_count = 0
  stride_x_start = 0.0
  stride_time_start = 0.0

  print("Starting simulation loop...")

  with torch.inference_mode():
    while True:
      steps_count += 1
      action = policy(state)
      state, _, reward, done, info = env.step(action)

      # Extract current data
      current_time = env.robot.time_since_reset.item()
      velocity = torch.norm(env.robot.base_vel).item()
      current_contacts = env.robot.foot_contacts[0].tolist()

      # Collect data for COT calculation
      current_joint_torques = env.robot.motor_torques[0].cpu().numpy()  # Shape: (12,)
      current_joint_velocities = env.robot.motor_velocities[0].cpu().numpy()  # Shape: (12,)

      # Store in pre-allocated buffers (overwrite, don't append)
      if stride_step_count < max_stride_steps:
        stride_torques[:, stride_step_count] = current_joint_torques
        stride_velocities[:, stride_step_count] = current_joint_velocities
        stride_step_count += 1


      # Extract stride length from environment
      stride_length = env._jumping_distance[0,0].item() if torch.is_tensor(env._jumping_distance[0]) else float(env._jumping_distance)
      # print(f"Current Stride Length: {stride_length:.3f}m, Time: {current_time:.2f}s")


      # Detect stride events
      is_stride_event = False
      if (current_time - prev_skip_time) > time_to_skip:
        if current_contacts == foot_fall_pattern:
          is_stride_event = True
          prev_skip_time = current_time
          # print(f">>> STRIDE EVENT detected at time {current_time:.2f}s <<<")

          # Calculate COT for completed stride
          if stride_step_count > 0:
            x_end = env.robot.base_position[0, 0].item()

            stride_velocity, stride_cot = calculate_cot_per_stride(
              stride_torques[:, :stride_step_count],      # Only use filled portion
              stride_velocities[:, :stride_step_count],   # Only use filled portion
              env._config.env_dt,                         # Time step
              stride_x_start,                             # X start
              x_end,                                      # X end
              stride_time_start,                          # Time start
              current_time,                               # Time end
              env.robot.mass                              # Mass 12.032528 Kg
            )

            # print(f"Stride Velocity: {stride_velocity:.3f} m/s, COT: {stride_cot:.4f}")
            # print(f"Mass: {env.robot.mass:3f} Kg")

            # Optional: Filter out invalid strides (like MATLAB does with COT < 20)
            if stride_cot < 20:
              if cot_data_count < max_strides:
                cot_data_arrays['velocity'][cot_data_count] = stride_velocity
                cot_data_arrays['cot'][cot_data_count] = stride_cot
                # cot_data_arrays['time'][cot_data_count] = current_time
                # cot_data_arrays['stride_length'][cot_data_count] = stride_length
                cot_data_count += 1

              if plotter is not None:
                plotter.add_cot_data(stride_velocity, stride_cot)


          # Reset for next stride (overwrite from beginning)
          stride_step_count = 0
          stride_x_start = env.robot.base_position[0, 0].item()
          stride_time_start = current_time

      # Update real-time plot (if enabled) - CALL EVERY STEP
      if plotter is not None:
        plotter.add_data(
          time_val=current_time,
          velocity=velocity,
          stride_length=stride_length,
          is_stride_event=is_stride_event
        )

      total_reward += reward
      logs.extend(info["logs"])

      if done.any():
        print(info["episode"])
        break

  end_time = time.time()
  elapsed = end_time - start_time

  print(f"\n{'='*60}")
  print(f"Simulation Complete!")
  print(f"{'='*60}")
  print(f"Total reward: {total_reward.item():.2f}")
  print(f"Total steps: {steps_count}")
  print(f"Time elapsed: {elapsed:.2f}s")
  print(f"Steps per second: {steps_count / elapsed:.1f} Hz")
  print(f"{'='*60}\n")

  # Save COT data to CSV efficiently
  if cot_data_count > 0:
    # Trim arrays to actual data size
    cot_output_path = os.path.join(root_path, f"cot_data_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv")
    # Stack arrays and save (most efficient for numpy)
    data_to_save = np.column_stack([
      cot_data_arrays['velocity'][:cot_data_count],
      cot_data_arrays['cot'][:cot_data_count],
      # cot_data_arrays['time'][:cot_data_count],
      # cot_data_arrays['stride_length'][:cot_data_count]
    ])
    np.savetxt(cot_output_path, data_to_save, 
               delimiter=',', 
               header='velocity,cot',#,time,stride_length',
               comments='',
               fmt='%.6f')

    print(f"COT data saved to: {cot_output_path}")
    print(f"Total strides recorded: {cot_data_count}")

  if plotter is not None:
    print("Plot window is open. Press Enter to close and exit...")
    input()
    plotter.close()

  if FLAGS.use_real_robot or FLAGS.save_traj:
    mode = "real" if FLAGS.use_real_robot else "sim"
    output_dir = f"eval_{mode}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.pkl"
    output_path = os.path.join(root_path, output_dir)

    with open(output_path, "wb") as fh:
      pickle.dump(logs, fh)
    print(f"Data logged to: {output_path}")


if __name__ == "__main__":
  app.run(main)
