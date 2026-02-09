"""Policy outputs desired CoM speed for Go1 to track the desired speed."""
from absl import logging

import itertools
import time
from typing import Sequence, Tuple

from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import to_torch
import ml_collections
import numpy as np
import torch

from src.configs.defaults import sim_config
from src.controllers import phase_gait_generator
from src.controllers import qp_torque_optimizer
from src.controllers import raibert_swing_leg_controller
from src.envs import go1_rewards
from src.robots import go1, go1_robot
from src.robots.motors import MotorControlMode, MotorCommand


# @torch.jit.script
def torch_rand_float(lower, upper, shape: Sequence[int], device: str):
  return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def gravity_frame_to_world_frame(robot_yaw, gravity_frame_vec):
  cos_yaw = torch.cos(robot_yaw)
  sin_yaw = torch.sin(robot_yaw)
  world_frame_vec = torch.clone(gravity_frame_vec)
  world_frame_vec[:, 0] = (cos_yaw * gravity_frame_vec[:, 0] -
                           sin_yaw * gravity_frame_vec[:, 1])
  world_frame_vec[:, 1] = (sin_yaw * gravity_frame_vec[:, 0] +
                           cos_yaw * gravity_frame_vec[:, 1])
  return world_frame_vec


@torch.jit.script
def world_frame_to_gravity_frame(robot_yaw, world_frame_vec):
  cos_yaw = torch.cos(robot_yaw)
  sin_yaw = torch.sin(robot_yaw)
  gravity_frame_vec = torch.clone(world_frame_vec)
  gravity_frame_vec[:, 0] = (cos_yaw * world_frame_vec[:, 0] +
                             sin_yaw * world_frame_vec[:, 1])
  gravity_frame_vec[:, 1] = (sin_yaw * world_frame_vec[:, 0] -
                             cos_yaw * world_frame_vec[:, 1])
  return gravity_frame_vec


def create_sim(sim_conf):
  gym = gymapi.acquire_gym()
  _, sim_device_id = gymutil.parse_device_str(sim_conf.sim_device)
  if sim_conf.show_gui:
    graphics_device_id = sim_device_id
  else:
    graphics_device_id = -1

  sim = gym.create_sim(sim_device_id, graphics_device_id,
                       sim_conf.physics_engine, sim_conf.sim_params)

  if sim_conf.show_gui:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "QUIT")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_V,
                                        "toggle_viewer_sync")
  else:
    viewer = None
  return gym, sim, viewer


class JumpEnv:

  def __init__(self,
               num_envs: int,
               config: ml_collections.ConfigDict(),
               device: str = "cuda",
               show_gui: bool = False,
               use_real_robot: bool = False):
    self._num_envs = num_envs
    self._device = device
    self._show_gui = show_gui
    self._config = config
    self._use_real_robot = use_real_robot
    self._use_raibert_controller = config.get('use_raibert_controller', True)
    self._jumping_distance_schedule = config.get('jumping_distance_schedule',
                                                 None)
    if self._jumping_distance_schedule is not None:
      self._jumping_distance_schedule = itertools.cycle(
          self._jumping_distance_schedule)
    with self._config.unlocked():
      self._config.goal_lb = to_torch(self._config.goal_lb, device=self._device)
      self._config.goal_ub = to_torch(self._config.goal_ub, device=self._device)
      if self._config.get('observation_noise', None) is not None:
        self._config.observation_noise = to_torch(
            self._config.observation_noise, device=self._device)
#################################
      self.velocity_lb = to_torch(self._config.velocity_lb, device=self._device)
      self.velocity_ub = to_torch(self._config.velocity_ub, device=self._device)

    self._desired_velocity = torch.zeros(self._num_envs, 3, device=self._device)
#################################

    # Set up robot and controller
    use_gpu = ("cuda" in device)
    self._sim_conf = sim_config.get_config(
        use_gpu=use_gpu,
        show_gui=show_gui,
        use_penetrating_contact=self._config.get('use_penetrating_contact',
                                                 False))
    self._gym, self._sim, self._viewer = create_sim(self._sim_conf)
    self._create_terrain()
    self._init_positions = self._compute_init_positions()
    if self._use_real_robot:
      robot_class = go1_robot.Go1Robot
    else:
      robot_class = go1.Go1
    self._robot = robot_class(
        num_envs=self._num_envs,
        init_positions=self._init_positions,
        sim=self._sim,
        viewer=self._viewer,
        sim_config=self._sim_conf,
        motor_control_mode=MotorControlMode.HYBRID,
        motor_torque_delay_steps=self._config.get('motor_torque_delay_steps',
                                                  0))
    strength_ratios = self._config.get('motor_strength_ratios', 0.7)
    if isinstance(strength_ratios, Sequence) and len(strength_ratios) == 2:
      ratios = torch_rand_float(lower=to_torch([strength_ratios[0]],
                                               device=self._device),
                                upper=to_torch([strength_ratios[1]],
                                               device=self._device),
                                shape=(self._num_envs, 3),
                                device=self._device)
      # Use the same ratio for all ab/ad motors, all hip motors, all knee motors
      ratios = torch.concatenate((ratios, ratios, ratios, ratios), dim=1)
      self._robot.motor_group.strength_ratios = ratios
    else:
      self._robot.motor_group.strength_ratios = strength_ratios

    # Need to set frictions twice to make it work on GPU... 😂
    self._robot.set_foot_frictions(0.01)
    self._robot.set_foot_frictions(self._config.get('foot_friction', 1.))
    self._gait_generator = phase_gait_generator.PhaseGaitGenerator(
        self._robot, self._config.gait)

    # Only create swing leg controller and torque optimizer if using Raibert
    if self._use_raibert_controller:
      self._swing_leg_controller = raibert_swing_leg_controller.RaibertSwingLegController(
          self._robot,
          self._gait_generator,
          foot_height=self._config.get('swing_foot_height', 0.),
          foot_landing_clearance=self._config.get('swing_foot_landing_clearance',
                                                  0.))
      self._torque_optimizer = qp_torque_optimizer.QPTorqueOptimizer(
          self._robot,
          base_position_kp=self._config.get('base_position_kp',
                                            np.array([0., 0., 50.])),
          base_position_kd=self._config.get('base_position_kd',
                                            np.array([10., 10., 10.])),
          base_orientation_kp=self._config.get('base_orientation_kp',
                                               np.array([50., 50., 0.])),
          base_orientation_kd=self._config.get('base_orientation_kd',
                                               np.array([10., 10., 10.])),
          weight_ddq=self._config.get('qp_weight_ddq',
                                      np.diag([20.0, 20.0, 5.0, 1.0, 1.0, .2])),
          foot_friction_coef=self._config.get('qp_foot_friction_coef', 0.7),
          clip_grf=self._config.get('clip_grf_in_sim') or self._use_real_robot,
          body_inertia=self._config.get('qp_body_inertia',
                                        np.array([0.14, 0.35, 0.35]) * 0.5),
          use_full_qp=self._config.get('use_full_qp', False))
    else:
      # Direct joint control mode: set up default standing pose
      # Go1 default standing: [ab/ad, hip, knee] x 4 legs
      # Order: FR, FL, RR, RL
      self._default_joint_positions = to_torch(
          self._config.get('default_joint_positions',
                           [0.0, 0.9, -1.8] * 4),
          device=self._device)
      self._direct_control_kp = self._config.get('direct_control_kp', 30.0)
      self._direct_control_kd = self._config.get('direct_control_kd', 1.0)

      # Dummy references so reward functions that access torque_optimizer
      # attributes don't crash. These are only used in the Raibert path
      # but some reward functions might reference them.
      self._desired_acc = torch.zeros((self._num_envs, 6), device=self._device)
      self._solved_acc = torch.zeros((self._num_envs, 6), device=self._device)
      self._qp_cost = torch.zeros(self._num_envs, device=self._device)
      self._num_clips = torch.zeros(self._num_envs, device=self._device)

    self._steps_count = torch.zeros(self._num_envs, device=self._device)
    self._init_yaw = torch.zeros(self._num_envs, device=self._device)
    self._episode_length = self._config.episode_length_s / self._config.env_dt
    self._construct_observation_and_action_space()
    if self._config.get("observe_heights", False):
      self._height_points, self._num_height_points = self._compute_height_points(
      )
    self._obs_buf = None
    self._privileged_obs_buf = None
    self._desired_landing_position = torch.zeros((self._num_envs, 3),
                                                 device=self._device,
                                                 dtype=torch.float)
    self._cycle_count = torch.zeros(self._num_envs, device=self._device)
    self._jumping_distance = torch.zeros((self._num_envs, 2),
                                         device=self._device)
    self._resample_command(torch.arange(self._num_envs, device=self._device))

    self._rewards = go1_rewards.Go1Rewards(self)
    self._prepare_rewards()
    self._extras = dict()

    # Running a few steps with dummy commands to ensure JIT compilation
    if self._use_raibert_controller:
      if self._num_envs == 1 and self._use_real_robot:
        for state in range(16):
          desired_contact_state = torch.tensor(
              [[(state & (1 << i)) != 0 for i in range(4)]],
              dtype=torch.bool,
              device=self._device)
          for _ in range(3):
            self._gait_generator.update()
            self._swing_leg_controller.update(self._desired_velocity)
            desired_foot_positions = self._swing_leg_controller.desired_foot_positions
            self._torque_optimizer.get_action(
                desired_contact_state, swing_foot_position=desired_foot_positions)

  def _create_terrain(self):
    """Creates terrains.

    Note that we set the friction coefficient to all 0 here. This is because
    Isaac seems to pick the larger friction out of a contact pair as the
    actual friction coefficient. We will set the corresponding friction coef
    in robot friction.
    """
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = 1.
    plane_params.dynamic_friction = 1.
    plane_params.restitution = 0.
    self._gym.add_ground(self._sim, plane_params)
    self._terrain = None

  def _compute_init_positions(self):
    init_positions = torch.zeros((self._num_envs, 3), device=self._device)

    num_cols = int(np.sqrt(self._num_envs))
    distance = 1.
    for idx in range(self._num_envs):
      init_positions[idx, 0] = idx // num_cols * distance
      init_positions[idx, 1] = idx % num_cols * distance
      init_positions[idx, 2] = 0.268
    return to_torch(init_positions, device=self._device)

  def _construct_observation_and_action_space(self):
    if self._use_raibert_controller:
      # === ORIGINAL CODE PATH (unchanged) ===
      robot_lb = to_torch( 
          [0., -3.14, -3.14, -4., -4., -10., -3.14, -3.14, -3.14] +
          [-0.5, -0.5, -0.4] * 4,
          device=self._device)
      robot_ub = to_torch([0.6, 3.14, 3.14, 4., 4., 10., 3.14, 3.14, 3.14] +
                          [0.5, 0.5, 0.] * 4,
                          device=self._device)

      task_lb = to_torch([-2., -2., -1., -1., -1.], device=self._device)
      task_ub = to_torch([2., 2., 1., 1., 1.], device=self._device)

      vel_lb = torch.full((2,), self.velocity_lb, device=self._device)
      vel_ub = torch.full((2,), self.velocity_ub, device=self._device)

      self._observation_lb = torch.concatenate((task_lb, vel_lb, robot_lb))
      self._observation_ub = torch.concatenate((task_ub, vel_ub, robot_ub))

      if self._config.get("observe_heights", False):
          num_heightpoints = len(self._config.measured_points_x) * len(
              self._config.measured_points_y)
          self._observation_lb = torch.concatenate(
              (self._observation_lb,
               torch.zeros(num_heightpoints, device=self._device) - 3))
          self._observation_ub = torch.concatenate(
              (self._observation_ub,
               torch.zeros(num_heightpoints, device=self._device) + 3))
      
      # Handle action bounds
      if self._config.get('learn_raibert_alpha', False):
          base_action_lb = to_torch(self._config.action_lb, device=self._device)
          base_action_ub = to_torch(self._config.action_ub, device=self._device)
          
          alpha_lb = torch.tensor([0.5], device=self._device)
          alpha_ub = torch.tensor([2.0], device=self._device)
          
          self._action_lb = torch.concatenate([base_action_lb, alpha_lb])
          self._action_ub = torch.concatenate([base_action_ub, alpha_ub])
      else:
          self._action_lb = to_torch(self._config.action_lb, device=self._device)
          self._action_ub = to_torch(self._config.action_ub, device=self._device)

    else:
      # === DIRECT JOINT CONTROL PATH ===
      # Observation: [distance_to_goal(3), phase(2), velocity_cmd(2),
      #               base_height(1), base_rpy(2), base_vel(3), base_ang_vel(3),
      #               desired_contact_state(4),
      #               motor_positions(12), motor_velocities(12),
      #               foot_positions_in_base_frame(12)]
      # = 3 + 2 + 2 + 1 + 2 + 3 + 3 + 4 + 12 + 12 + 12 = 56

      task_lb = to_torch([-2., -2., -1., -1., -1.], device=self._device)  # dist_to_goal(3) + phase(2)
      task_ub = to_torch([2., 2., 1., 1., 1.], device=self._device)

      vel_lb = torch.full((2,), self.velocity_lb, device=self._device)
      vel_ub = torch.full((2,), self.velocity_ub, device=self._device)

      # base_height(1), roll(1), pitch(1), vx(1), vy(1), vz(1), wx(1), wy(1), wz(1)
      base_state_lb = to_torch(
          [0., -3.14, -3.14, -4., -4., -10., -3.14, -3.14, -3.14],
          device=self._device)
      base_state_ub = to_torch(
          [0.6, 3.14, 3.14, 4., 4., 10., 3.14, 3.14, 3.14],
          device=self._device)

      # desired_contact_state(4): binary 0/1
      contact_lb = torch.zeros(4, device=self._device)
      contact_ub = torch.ones(4, device=self._device)

      # motor_positions(12): Go1 joint limits
      # ab/ad: [-1.047, 1.047], hip: [-0.663, 2.966], knee: [-2.721, -0.837]
      motor_pos_lb = to_torch(
          [-1.047, -0.663, -2.721] * 4, device=self._device)
      motor_pos_ub = to_torch(
          [1.047, 2.966, -0.837] * 4, device=self._device)

      # motor_velocities(12)
      motor_vel_lb = torch.full((12,), -20., device=self._device)
      motor_vel_ub = torch.full((12,), 20., device=self._device)

      # foot_positions_in_base_frame(12)
      foot_pos_lb = to_torch([-0.5, -0.5, -0.4] * 4, device=self._device)
      foot_pos_ub = to_torch([0.5, 0.5, 0.] * 4, device=self._device)

      self._observation_lb = torch.concatenate((
          task_lb, vel_lb, base_state_lb, contact_lb,
          motor_pos_lb, motor_vel_lb, foot_pos_lb))
      self._observation_ub = torch.concatenate((
          task_ub, vel_ub, base_state_ub, contact_ub,
          motor_pos_ub, motor_vel_ub, foot_pos_ub))

      # Action: 12 joint position offsets from default standing pose
      # Offset range per joint type:
      #   ab/ad: [-0.5, 0.5] rad
      #   hip:   [-1.0, 1.0] rad  (needs larger range for swing)
      #   knee:  [-1.0, 1.0] rad  (needs larger range for swing)
      action_offset_lb = self._config.get(
          'direct_action_lb', [-0.5, -1.0, -1.0] * 4)
      action_offset_ub = self._config.get(
          'direct_action_ub', [0.5, 1.0, 1.0] * 4)
      self._action_lb = to_torch(action_offset_lb, device=self._device)
      self._action_ub = to_torch(action_offset_ub, device=self._device)

  def _prepare_rewards(self):
    self._reward_names, self._reward_fns, self._reward_scales = [], [], []
    self._episode_sums = dict()
    for name, scale in self._config.rewards:
      self._reward_names.append(name)
      self._reward_fns.append(getattr(self._rewards, name + '_reward'))
      self._reward_scales.append(scale)
      self._episode_sums[name] = torch.zeros(self._num_envs,
                                             device=self._device)

    self._terminal_reward_names, self._terminal_reward_fns, self._terminal_reward_scales = [], [], []
    for name, scale in self._config.terminal_rewards:
      self._terminal_reward_names.append(name)
      self._terminal_reward_fns.append(getattr(self._rewards, name + '_reward'))
      self._terminal_reward_scales.append(scale)
      self._episode_sums[name] = torch.zeros(self._num_envs,
                                             device=self._device)

  def reset(self) -> torch.Tensor:
    return self.reset_idx(torch.arange(self._num_envs, device=self._device))

  def _split_action(self, action):
      """Split action into components. Only used in Raibert controller path."""
      # Extract alpha FIRST if learning Raibert gain (it's the last element)
      alpha = None
      if self._config.get('learn_raibert_alpha', False):
          alpha = action[:, -1:]  # Last element is alpha
          action = action[:, :-1]  # Remove alpha from main action
      
      # Now process the remaining action components in order
      gait_action = None
      if self._config.get('include_gait_action', False):
          gait_action = action[:, :1]
          action = action[:, 1:]
      
      foot_action = None
      if self._config.get('include_foot_action', False):
          if self._config.get('mirror_foot_action', False):
              foot_action = action[:, -6:].reshape((-1, 2, 3))
              foot_action = torch.stack([
                  foot_action[:, 0],
                  foot_action[:, 0],
                  foot_action[:, 1],
                  foot_action[:, 1],
              ], dim=1)
              action = action[:, :-6]
          else:
              foot_action = action[:, -12:].reshape((-1, 4, 3))
              action = action[:, :-12]
      
      com_action = action
      return gait_action, com_action, foot_action, alpha

  def reset_idx(self, env_ids) -> torch.Tensor:
    # Aggregate rewards
    self._extras["time_outs"] = self._episode_terminated()
    if env_ids.shape[0] > 0:
      self._extras["episode"] = {}
      self._extras["episode"]["cycle_count"] = torch.mean(
          self._gait_generator.true_phase[env_ids]) / (2 * torch.pi)

      self._obs_buf = self._get_observations()
      self._privileged_obs_buf = self._get_privileged_observations()

      for reward_name in self._episode_sums.keys():
        if reward_name in self._reward_names:
          if self._config.get('normalize_reward_by_phase', False):
            self._extras["episode"]["reward_{}".format(
                reward_name)] = torch.mean(
                    self._episode_sums[reward_name][env_ids] /
                    (self._gait_generator.true_phase[env_ids] / (2 * torch.pi)))
          else:
            # Normalize by time
            self._extras["episode"]["reward_{}".format(
                reward_name)] = torch.mean(
                    self._episode_sums[reward_name][env_ids] /
                    (self._steps_count[env_ids] * (self._config.env_dt)))

        if reward_name in self._terminal_reward_names:
          self._extras["episode"]["reward_{}".format(reward_name)] = torch.mean(
              self._episode_sums[reward_name][env_ids] /
              (self._cycle_count[env_ids].clip(min=1)))

        self._episode_sums[reward_name][env_ids] = 0

      r = torch.rand(env_ids.shape[0], device=self.device)  # sample N random numbers in [0, 1)
      desired_velocity_x = self.velocity_lb + r * (self.velocity_ub - self.velocity_lb)
      self._desired_velocity[env_ids, 0]= desired_velocity_x
      self._desired_velocity[env_ids, 1:] = 0

      self._steps_count[env_ids] = 0
      self._cycle_count[env_ids] = 0
      self._init_yaw[env_ids] = self._robot.base_orientation_rpy[env_ids, 2]
      self._robot.reset_idx(env_ids)
      if self._use_raibert_controller:
        self._swing_leg_controller.reset_idx(env_ids)
      self._gait_generator.reset_idx(env_ids)
      self._resample_command(env_ids)

    return self._obs_buf, self._privileged_obs_buf

  def _step_direct_joint_control(self, action: torch.Tensor):
    """Step function for direct joint control (no Raibert controller).
    
    The NN outputs 12 joint position offsets that are added to the default
    standing pose. A PD controller at the motor level tracks these targets.
    The gait generator still runs to provide phase information for 
    observations and rewards.
    """
    self._last_action = torch.clone(action)
    action = torch.clip(action, self._action_lb, self._action_ub)
    sum_reward = torch.zeros(self._num_envs, device=self._device)
    dones = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
    self._steps_count += 1
    logs = []

    # Compute desired joint positions: default pose + NN offsets
    desired_motor_positions = self._default_joint_positions.unsqueeze(0) + action

    # Clamp to joint limits for safety
    # Go1 joint limits: ab/ad [-1.047, 1.047], hip [-0.663, 2.966], knee [-2.721, -0.837]
    joint_lower = to_torch([-1.047, -0.663, -2.721] * 4, device=self._device)
    joint_upper = to_torch([1.047, 2.966, -0.837] * 4, device=self._device)
    desired_motor_positions = torch.clip(desired_motor_positions,
                                         joint_lower, joint_upper)

    # Create motor command: pure position control via PD
    motor_action = MotorCommand(
        desired_position=desired_motor_positions,
        kp=torch.ones(self._num_envs, 12, device=self._device) * self._direct_control_kp,
        desired_velocity=torch.zeros(self._num_envs, 12, device=self._device),
        kd=torch.ones(self._num_envs, 12, device=self._device) * self._direct_control_kd,
        desired_extra_torque=torch.zeros(self._num_envs, 12, device=self._device))

    for step in range(
        max(int(self._config.env_dt / self._robot.control_timestep), 1)):
      # Gait generator still runs - needed for obs and rewards
      self._gait_generator.update()

      if self._use_real_robot:
        self._robot.state_estimator.update_foot_contact(
            self._gait_generator.desired_contact_state)
        self._robot.update_desired_foot_contact(
            self._gait_generator.desired_contact_state)

      logs.append(
          dict(timestamp=self._robot.time_since_reset,
               base_position=torch.clone(self._robot.base_position),
               base_orientation_rpy=torch.clone(
                   self._robot.base_orientation_rpy),
               base_velocity=torch.clone(self._robot.base_velocity_body_frame),
               base_angular_velocity=torch.clone(
                   self._robot.base_angular_velocity_body_frame),
               motor_positions=torch.clone(self._robot.motor_positions),
               motor_velocities=torch.clone(self._robot.motor_velocities),
               motor_action=motor_action,
               motor_torques=self._robot.motor_torques,
               num_clips=self._num_clips,
               foot_contact_state=self._gait_generator.desired_contact_state,
               foot_contact_force=self._robot.foot_contact_forces,
               desired_motor_positions=desired_motor_positions,
               foot_positions_in_base_frame=self._robot.foot_positions_in_base_frame,
               env_action=action,
               env_obs=torch.clone(self._obs_buf) if self._obs_buf is not None else None))
      if self._use_real_robot:
        logs[-1]["base_acc"] = np.array(self._robot.raw_state.imu.accelerometer)

      self._robot.step(motor_action)

      self._obs_buf = self._get_observations()
      self._privileged_obs_buf = self.get_privileged_observations()
      rewards = self.get_reward()
      dones = torch.logical_or(dones, self._is_done())
      sum_reward += rewards * torch.logical_not(dones)

    self._extras["logs"] = logs
    # Resample commands
    new_cycle_count = (self._gait_generator.true_phase / (2 * torch.pi)).long()
    finished_cycle = new_cycle_count > self._cycle_count
    env_ids_to_resample = finished_cycle.nonzero(as_tuple=False).flatten()
    self._cycle_count = new_cycle_count

    is_terminal = torch.logical_or(finished_cycle, dones)
    if is_terminal.any():
      sum_reward += self.get_terminal_reward(is_terminal, dones)
    self._resample_command(env_ids_to_resample)
    if not self._use_real_robot:
      self.reset_idx(dones.nonzero(as_tuple=False).flatten())

    if self._show_gui:
      self._robot.render()
    return self._obs_buf, self._privileged_obs_buf, sum_reward, dones, self._extras

  def step(self, action: torch.Tensor):
    if not self._use_raibert_controller:
      return self._step_direct_joint_control(action)

    # === ORIGINAL RAIBERT CONTROLLER PATH (unchanged) ===
    self._last_action = torch.clone(action)
    action = torch.clip(action, self._action_lb, self._action_ub)
    sum_reward = torch.zeros(self._num_envs, device=self._device)
    dones = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
    self._steps_count += 1
    logs = []

    zero = torch.zeros(self._num_envs, device=self._device)
    gait_action, com_action, foot_action, alpha = self._split_action(action)

    if alpha is not None:
        self._swing_leg_controller.set_kp_alpha(alpha)

    desired_linear_vel_z = (com_action[:, 2] -
                            self._torque_optimizer.desired_base_position[:, 2]
                           ) / self._config.env_dt
    desired_linear_vel_z = desired_linear_vel_z.clip(min=-0., max=0.)
    desired_ang_vel_y = (
        com_action[:, 4] -
        self._torque_optimizer.desired_base_orientation_rpy[:, 1]
    ) / self._config.env_dt
    desired_ang_vel_y = desired_ang_vel_y.clip(min=-0., max=0.)

    for step in range(
        max(int(self._config.env_dt / self._robot.control_timestep), 1)):
      self._gait_generator.update()
      self._swing_leg_controller.update()

      if self._use_real_robot:
        self._robot.state_estimator.update_foot_contact(
            self._gait_generator.desired_contact_state)  # pytype: disable=attribute-error
        self._robot.update_desired_foot_contact(
            self._gait_generator.desired_contact_state)  # pytype: disable=attribute-error

      if gait_action is not None:
        self._gait_generator.stepping_frequency = gait_action[:, 0]

      # CoM pose action
      self._torque_optimizer.desired_base_position = torch.stack(
          (self._robot.base_position[:, 0], self._robot.base_position[:, 1],
           com_action[:, 0]),
          dim=1)
      self._torque_optimizer.desired_linear_velocity = torch.stack(
          (com_action[:, 1], com_action[:, 2] * 0, com_action[:, 3]), dim=1)
      self._torque_optimizer.desired_base_orientation_rpy = torch.stack(
          (com_action[:, 4] * 0, com_action[:, 5],
           self._robot.base_orientation_rpy[:, 2]),
          dim=1)

      if self._config.get('use_yaw_feedback', False):
        yaw_err = (self._init_yaw - self._robot.base_orientation_rpy[:, 2])
        yaw_err = torch.remainder(yaw_err + 3 * torch.pi,
                                  2 * torch.pi) - torch.pi
        desired_yaw_rate = 1 * yaw_err
        self._torque_optimizer.desired_angular_velocity = torch.stack(
            (zero, com_action[:, 6], desired_yaw_rate), dim=1)
      else:
        self._torque_optimizer.desired_angular_velocity = torch.stack(
            (zero, com_action[:, 6], com_action[:, 7] * 0), dim=1)

      desired_foot_positions = self._swing_leg_controller.desired_foot_positions
      if foot_action is not None:
        base_yaw = self._robot.base_orientation_rpy[:, 2]
        cos_yaw = torch.cos(base_yaw)[:, None]
        sin_yaw = torch.sin(base_yaw)[:, None]
        foot_action_world = torch.clone(foot_action)
        foot_action_world[:, :, 0] = (cos_yaw * foot_action[:, :, 0] -
                                      sin_yaw * foot_action[:, :, 1])
        foot_action_world[:, :, 1] = (sin_yaw * foot_action[:, :, 0] +
                                      cos_yaw * foot_action[:, :, 1])
        desired_foot_positions += foot_action_world

      # DEBUG: View projected rectagles on the ground
      # if self._show_gui and step == 0:  # Only draw once per env.step(), not every control loop iteration
      if (self._show_gui or self._record_video) and step == 0:
        self._gym.clear_lines(self._viewer)
        
        # Draw desired foot landing positions
        env_id = 0  # Only visualize first environment
        for foot_id in range(4):
            foot_pos = desired_foot_positions[env_id, foot_id].cpu().numpy()
            
            # Different color for each foot
            colors = [
                [1, 0, 0],  # Front-left: Red
                [0, 1, 0],  # Front-right: Green
                [0, 0, 1],  # Rear-left: Blue
                [1, 1, 0]   # Rear-right: Yellow
            ]
            
            # Draw box at foot position with 0.35m length x 0.6m width
            self._draw_box(self._gym, self._viewer, 
                    center=(foot_pos[0], foot_pos[1], foot_pos[2], 0.175, 0.3),
                    color=colors[foot_id])

      motor_action, self._desired_acc, self._solved_acc, self._qp_cost, self._num_clips = self._torque_optimizer.get_action(
          self._gait_generator.desired_contact_state,
          swing_foot_position=desired_foot_positions)

      logs.append(
          dict(timestamp=self._robot.time_since_reset,
               base_position=torch.clone(self._robot.base_position),
               base_orientation_rpy=torch.clone(
                   self._robot.base_orientation_rpy),
               base_velocity=torch.clone(self._robot.base_velocity_body_frame),
               base_angular_velocity=torch.clone(
                   self._robot.base_angular_velocity_body_frame),
               motor_positions=torch.clone(self._robot.motor_positions),
               motor_velocities=torch.clone(self._robot.motor_velocities),
               motor_action=motor_action,
               motor_torques=self._robot.motor_torques,
               num_clips=self._num_clips,
               foot_contact_state=self._gait_generator.desired_contact_state, foot_contact_force=self._robot.foot_contact_forces, desired_swing_foot_position=desired_foot_positions, desired_acc_body_frame=self._desired_acc,
               solved_acc_body_frame=self._solved_acc,
               foot_positions_in_base_frame=self._robot.
               foot_positions_in_base_frame,
               env_action=action,
               env_obs=torch.clone(self._obs_buf)))
      if self._use_real_robot:
        logs[-1]["base_acc"] = np.array(self._robot.raw_state.imu.accelerometer)  # pytype: disable=attribute-error

      self._robot.step(motor_action)

      self._obs_buf = self._get_observations()
      self._privileged_obs_buf = self.get_privileged_observations()
      rewards = self.get_reward()
      dones = torch.logical_or(dones, self._is_done())
      sum_reward += rewards * torch.logical_not(dones)

    self._extras["logs"] = logs
    # Resample commands
    new_cycle_count = (self._gait_generator.true_phase / (2 * torch.pi)).long()
    finished_cycle = new_cycle_count > self._cycle_count
    env_ids_to_resample = finished_cycle.nonzero(as_tuple=False).flatten()
    self._cycle_count = new_cycle_count

    is_terminal = torch.logical_or(finished_cycle, dones)
    if is_terminal.any():
      sum_reward += self.get_terminal_reward(is_terminal, dones)
    self._resample_command(env_ids_to_resample)
    if not self._use_real_robot:
      self.reset_idx(dones.nonzero(as_tuple=False).flatten())

    if self._show_gui:
      self._robot.render()
    return self._obs_buf, self._privileged_obs_buf, sum_reward, dones, self._extras

  def _resample_command(self, env_ids):
    if env_ids.shape[0] == 0:
      return

    if self._num_envs == 1 and self._jumping_distance_schedule is not None:
      self._jumping_distance = torch.tensor(
          [[next(self._jumping_distance_schedule), 0]], device=self._device)
    else:
      self._jumping_distance[env_ids] = torch_rand_float(self._config.goal_lb,
                                                         self._config.goal_ub,
                                                         [env_ids.shape[0], 2],
                                                         device=self._device)

    self._desired_landing_position[env_ids, :2] = self._robot.base_position[
        env_ids, :2] + gravity_frame_to_world_frame(
            self._robot.base_orientation_rpy[env_ids, 2],
            self._jumping_distance[env_ids])
    self._desired_landing_position[env_ids, 2] = 0.268

  def _get_observations(self):
    if not self._use_raibert_controller:
      return self._get_observations_direct()

    # === ORIGINAL OBSERVATION PATH (unchanged) ===
    distance_to_goal = self._desired_landing_position - self._robot.base_position_world

    distance_to_goal_local = world_frame_to_gravity_frame(
        self._robot.base_orientation_rpy[:, 2], distance_to_goal)
    phase_obs = torch.stack((
        torch.cos(self._gait_generator.true_phase),
        torch.sin(self._gait_generator.true_phase),
    ),
                            dim=1)
    velocity_command = self._desired_velocity[:, :2]

    robot_obs = torch.concatenate(
        (
            self._robot.base_position[:, 2:],  # Base height
            self._robot.base_orientation_rpy[:, 0:1] * 0,  # Base roll
            self._robot.base_orientation_rpy[:, 1:2],  # Base Pitch
            self._robot.base_velocity_body_frame[:, 0:1],
            self._robot.base_velocity_body_frame[:, 1:2] * 0,
            self._robot.base_velocity_body_frame[:, 2:3],  # Base velocity (z)
            self._robot.base_angular_velocity_body_frame[:, 0:1] * 0,
            self._robot.base_angular_velocity_body_frame[:, 1:2],
            self._robot.base_angular_velocity_body_frame[:,
                                                         2:3],  # Base yaw rate
            self._robot.foot_positions_in_base_frame.reshape(
                (self._num_envs, 12)),
        ),
        dim=1)
    obs = torch.concatenate((distance_to_goal_local, phase_obs, velocity_command, robot_obs),
                            dim=1)
    if self._config.get("observation_noise",
                        None) is not None and (not self._use_real_robot):
      obs += torch.randn_like(obs) * self._config.observation_noise
    return obs

  def _get_observations_direct(self):
    """Observations for direct joint control mode.
    
    Includes: distance_to_goal(3), phase(2), velocity_cmd(2),
              base_state(9), desired_contact_state(4),
              motor_positions(12), motor_velocities(12),
              foot_positions_in_base_frame(12)
    """
    distance_to_goal = self._desired_landing_position - self._robot.base_position_world
    distance_to_goal_local = world_frame_to_gravity_frame(
        self._robot.base_orientation_rpy[:, 2], distance_to_goal)

    phase_obs = torch.stack((
        torch.cos(self._gait_generator.true_phase),
        torch.sin(self._gait_generator.true_phase),
    ), dim=1)

    velocity_command = self._desired_velocity[:, :2]

    # Base state - same as original but WITHOUT zeroing out roll/vy/wx
    # The NN needs full state information for direct control
    base_state = torch.concatenate((
        self._robot.base_position[:, 2:],              # height (1)
        self._robot.base_orientation_rpy[:, 0:1],      # roll (1)
        self._robot.base_orientation_rpy[:, 1:2],      # pitch (1)
        self._robot.base_velocity_body_frame[:, 0:1],  # vx (1)
        self._robot.base_velocity_body_frame[:, 1:2],  # vy (1)
        self._robot.base_velocity_body_frame[:, 2:3],  # vz (1)
        self._robot.base_angular_velocity_body_frame[:, 0:1],  # wx (1)
        self._robot.base_angular_velocity_body_frame[:, 1:2],  # wy (1)
        self._robot.base_angular_velocity_body_frame[:, 2:3],  # wz (1)
    ), dim=1)

    # Desired contact state from gait generator (crucial for gait following)
    desired_contact = self._gait_generator.desired_contact_state.float()

    # Proprioceptive feedback (essential for direct joint control)
    motor_pos = self._robot.motor_positions
    motor_vel = self._robot.motor_velocities

    foot_pos = self._robot.foot_positions_in_base_frame.reshape(
        (self._num_envs, 12))

    obs = torch.concatenate((
        distance_to_goal_local,  # 3
        phase_obs,               # 2
        velocity_command,        # 2
        base_state,              # 9
        desired_contact,         # 4
        motor_pos,               # 12
        motor_vel,               # 12
        foot_pos,                # 12
    ), dim=1)

    if self._config.get("observation_noise",
                        None) is not None and (not self._use_real_robot):
      obs += torch.randn_like(obs) * self._config.observation_noise

    return obs

  def get_observations(self):
    return self._obs_buf

  def _get_privileged_observations(self):
    return None

  def get_privileged_observations(self):
    return self._privileged_obs_buf

  def get_reward(self):
    sum_reward = torch.zeros(self._num_envs, device=self._device)
    for idx in range(len(self._reward_names)):
      reward_name = self._reward_names[idx]
      reward_fn = self._reward_fns[idx]
      reward_scale = self._reward_scales[idx]
      reward_item = reward_scale * reward_fn()
      if self._config.get('normalize_reward_by_phase', False):
        reward_item *= self._gait_generator.stepping_frequency
      self._episode_sums[reward_name] += reward_item
      sum_reward += reward_item

    if self._config.clip_negative_reward:
      sum_reward = torch.clip(sum_reward, min=0)
    return sum_reward

  def get_terminal_reward(self, is_terminal, dones):
    early_term = torch.logical_and(
        dones, torch.logical_not(self._episode_terminated()))
    coef = torch.where(early_term, self._gait_generator.cycle_progress,
                       torch.ones_like(early_term))

    sum_reward = torch.zeros(self._num_envs, device=self._device)
    for idx in range(len(self._terminal_reward_names)):
      reward_name = self._terminal_reward_names[idx]
      reward_fn = self._terminal_reward_fns[idx]
      reward_scale = self._terminal_reward_scales[idx]
      reward_item = reward_scale * reward_fn() * is_terminal * coef
      self._episode_sums[reward_name] += reward_item
      sum_reward += reward_item

    if self._config.clip_negative_terminal_reward:
      sum_reward = torch.clip(sum_reward, min=0)
    return sum_reward

  def _episode_terminated(self):
    timeout = (self._steps_count >= self._episode_length)
    cycles_finished = (self._gait_generator.true_phase /
                       (2 * torch.pi)) > self._config.get('max_jumps', 1)
    return torch.logical_or(timeout, cycles_finished)

  def _is_done(self):
    is_unsafe = torch.logical_or(
        self._robot.projected_gravity[:, 2] < 0.5, self._robot.base_position[:,
                                                                             2]
        < self._config.get('terminate_on_height', 0.15))
    if self._config.get('terminate_on_body_contact', False):
      is_unsafe = torch.logical_or(is_unsafe, self._robot.has_body_contact)

    if self._config.get('terminate_on_limb_contact', False):
      limb_contact = torch.logical_or(self._robot.calf_contacts,
                                      self._robot.thigh_contacts)
      limb_contact = torch.sum(limb_contact, dim=1)
      is_unsafe = torch.logical_or(is_unsafe, limb_contact > 0)

    return torch.logical_or(self._episode_terminated(), is_unsafe)

  def _draw_box(self, gym, viewer, center=None, corners=None, color=[1, 0, 0]):
      """
      Draw a rectangular box on the ground plane (z=0)
      
      Args:
          gym: Isaac Gym instance
          viewer: Viewer instance
          center: tuple (x, y, z, half_length, half_width) - center point and dimensions
                  OR tuple (x, y, half_length, half_width) - assumes z=0
          corners: list of 4 corner points [(x1,y1,z1), (x2,y2,z2), ...] in absolute coords
                   If z is not provided, assumes z=0 for each corner
          color: RGB color values [r, g, b] from 0 to 1
      
      Usage:
          # Option 1: Specify center and dimensions
          draw_box(gym, viewer, center=(1.0, 2.0, 0.5, 0.03, 0.05))
          draw_box(gym, viewer, center=(1.0, 2.0, 0.03, 0.05))  # z=0 assumed
          
          # Option 2: Specify absolute corner positions
          draw_box(gym, viewer, corners=[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
          draw_box(gym, viewer, corners=[(0, 0), (1, 0), (1, 1), (0, 1)])  # z=0 assumed
      """
      
      if center is not None:
          # Parse center specification
          if len(center) == 5:
              x, y, z, half_length, half_width = center
          elif len(center) == 4:
              x, y, half_length, half_width = center
              z = 0
          else:
              raise ValueError("center must be (x, y, z, half_length, half_width) or (x, y, half_length, half_width)")
          
          # Calculate corners from center and dimensions
          corners = [
              [x - half_length, y - half_width, z],
              [x + half_length, y - half_width, z],
              [x + half_length, y + half_width, z],
              [x - half_length, y + half_width, z]
          ]
      
      elif corners is not None:
          # Use provided corners, ensure z=0 if not specified
          corners = [list(c) + [0] * (3 - len(c)) for c in corners]
          
          if len(corners) != 4:
              raise ValueError("Must provide exactly 4 corner points")
      
      else:
          raise ValueError("Must provide either 'center' or 'corners'")
      
      # Create line segments connecting the corners
      lines = []
      for i in range(4):
          start = corners[i]
          end = corners[(i + 1) % 4]
          lines.append(start + end)  # [x1, y1, z1, x2, y2, z2]
      
      # Convert to numpy arrays
      lines = np.array(lines, dtype=np.float32)
      colors = np.array([color] * 4, dtype=np.float32)
      
      # Add lines to viewer
      gym.add_lines(viewer, None, lines.shape[0], lines, colors)

  @property
  def device(self):
    return self._device

  @property
  def robot(self):
    return self._robot

  @property
  def gait_generator(self):
    return self._gait_generator

  @property
  def desired_landing_position(self):
    return self._desired_landing_position

  @property
  def action_space(self):
    return self._action_lb, self._action_ub

  @property
  def observation_space(self):
    return self._observation_lb, self._observation_ub

  @property
  def num_envs(self):
    return self._num_envs

  @property
  def num_obs(self):
    return self._observation_lb.shape[0]

  @property
  def num_privileged_obs(self):
    return None

  @property
  def num_actions(self):
    return self._action_lb.shape[0]

  @property
  def max_episode_length(self):
    return self._episode_length

  @property
  def episode_length_buf(self):
    return self._steps_count

  @episode_length_buf.setter
  def episode_length_buf(self, new_length: torch.Tensor):
    self._steps_count = to_torch(new_length, device=self._device)
    self._gait_generator._current_phase += 2 * torch.pi * (
        new_length / self.max_episode_length * self._config.get('max_jumps', 1)
        + 1)[:, None]
    self._cycle_count = (self._gait_generator.true_phase /
                         (2 * torch.pi)).long()

  @property
  def device(self):
    return self._device
