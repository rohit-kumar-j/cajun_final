"""Policy outputs desired CoM speed for Go2 to track the desired speed.

Changes from original (Step 1 migration):
  - Imports UnitreeQuad instead of Go1
  - Robot instantiation passes robot_config instead of motor_torque_delay_steps
  - init_height read from config.robot (with fallback) instead of hardcoded 0.268
  - All other logic unchanged
"""
from absl import logging

import itertools
import time
from typing import Sequence, Tuple

from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import to_torch
from loguru import logger
import ml_collections
import numpy as np
import torch

from src.configs.defaults import sim_config
from src.controllers import phase_gait_generator
from src.controllers import qp_torque_optimizer
from src.controllers import raibert_swing_leg_controller
from src.envs import go1_rewards
from src.robots import unitree_quad          # ← was: from src.robots import go1, go1_robot
from src.robots.motors import MotorControlMode
from src.utilities.config_utils import cfg, flush_warnings, REQUIRED


def torch_rand_float(lower, upper, shape: Sequence[int], device: str):
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def gravity_frame_to_world_frame(robot_yaw, gravity_frame_vec):
    cos_yaw = torch.cos(robot_yaw)
    sin_yaw = torch.sin(robot_yaw)
    world_frame_vec = torch.clone(gravity_frame_vec)
    world_frame_vec[:, 0] = cos_yaw * gravity_frame_vec[:, 0] - sin_yaw * gravity_frame_vec[:, 1]
    world_frame_vec[:, 1] = sin_yaw * gravity_frame_vec[:, 0] + cos_yaw * gravity_frame_vec[:, 1]
    return world_frame_vec


@torch.jit.script
def world_frame_to_gravity_frame(robot_yaw, world_frame_vec):
    cos_yaw = torch.cos(robot_yaw)
    sin_yaw = torch.sin(robot_yaw)
    gravity_frame_vec = torch.clone(world_frame_vec)
    gravity_frame_vec[:, 0] =  cos_yaw * world_frame_vec[:, 0] + sin_yaw * world_frame_vec[:, 1]
    gravity_frame_vec[:, 1] =  sin_yaw * world_frame_vec[:, 0] - cos_yaw * world_frame_vec[:, 1]
    return gravity_frame_vec


def create_sim(sim_conf):
    gym = gymapi.acquire_gym()
    _, sim_device_id = gymutil.parse_device_str(sim_conf.sim_device)
    graphics_device_id = sim_device_id if sim_conf.show_gui else -1
    sim = gym.create_sim(
        sim_device_id, graphics_device_id,
        sim_conf.physics_engine, sim_conf.sim_params
    )
    if sim_conf.show_gui:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "QUIT")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_V, "toggle_viewer_sync")
    else:
        viewer = None
    return gym, sim, viewer


class JumpEnv:

    def __init__(
        self,
        num_envs:        int,
        config:          ml_collections.ConfigDict,
        device:          str  = "cuda",
        show_gui:        bool = False,
        use_real_robot:  bool = False,
    ):
        self._num_envs       = num_envs
        self._device         = device
        self._show_gui       = show_gui
        self._config         = config
        self._use_real_robot = use_real_robot
        self._frame_callback= None
        self._capture_callback=None

        # ── Read all init-time config values via cfg() ────────────────────────
        # These are read ONCE at init. Hot-path methods use self._ cached attrs.
        _w: dict = {}
        C = "JumpEnv"  # caller label for warnings

        jumping_distance_schedule = cfg(config,     "jumping_distance_schedule", None,     C, _w)
        observation_noise_raw     = cfg(config,     "observation_noise",         None,     C, _w)
        use_penetrating_contact   = cfg(config.env, "use_penetrating_contact",   REQUIRED, C, _w)
        strength_ratios           = cfg(config.env, "motor_strength_ratios",     REQUIRED, C, _w)
        foot_friction             = cfg(config.env, "foot_friction",             REQUIRED, C, _w)
        observe_heights           = cfg(config.env, "observe_heights",           False   , C, _w)

        # ── Cache loop-path flags as self._ so hot methods never touch config ─
        self._init_height   = cfg(config.robot, "init_height",  REQUIRED, C, _w)

        self._learn_raibert_alpha    = cfg(config.env, "learn_raibert_alpha",  REQUIRED, C, _w)
        self._include_gait_action    = cfg(config.env, "include_gait_action",  REQUIRED, C, _w)
        self._include_foot_action    = cfg(config.env, "include_foot_action",  REQUIRED, C, _w)
        self._mirror_foot_action     = cfg(config.env, "mirror_foot_action",   REQUIRED, C, _w)
        self._use_yaw_feedback       = cfg(config.env, "use_yaw_feedback",     REQUIRED, C, _w)
        self._normalize_by_phase     = cfg(config.rewards, "normalize_reward_by_phase", REQUIRED, C, _w)
        self._max_jumps              = cfg(config.env, "max_jumps",                  REQUIRED, C, _w)
        self._terminate_on_height    = cfg(config.env, "terminate_on_height",       REQUIRED, C, _w)
        self._terminate_on_body      = cfg(config.env, "terminate_on_body_contact", REQUIRED, C, _w)
        self._terminate_on_limb      = cfg(config.env, "terminate_on_limb_contact", REQUIRED, C, _w)
        self._terminate_on_gravity_z = float(cfg(config.env, "terminate_on_gravity_z", REQUIRED,  C, _w))
        self._yaw_feedback_gain      = float(cfg(config.env, "yaw_feedback_gain",   REQUIRED,  C, _w))

        # These may be None, and there would be no problem
        self._has_obs_noise          = observation_noise_raw is not None
        self._observe_heights        = observe_heights

        # Write all warning flags into config.warnings
        flush_warnings(config, "JumpEnv", _w)

        # ── Tensors that come from config ─────────────────────────────────────
        self._jumping_distance_schedule = None
        if jumping_distance_schedule is not None:
            self._jumping_distance_schedule = itertools.cycle(jumping_distance_schedule)

        # goal_lb/ub and velocity_lb/ub now live in config.env
        _env_cfg = self._config.env
        with self._config.unlocked():
            if observation_noise_raw is not None:
                self._config.observation_noise = to_torch(
                    observation_noise_raw, device=self._device
                )
        with _env_cfg.unlocked():
            _env_cfg.goal_lb = to_torch(_env_cfg.goal_lb, device=self._device)
            _env_cfg.goal_ub = to_torch(_env_cfg.goal_ub, device=self._device)
        self.velocity_lb = to_torch(_env_cfg.velocity_lb, device=self._device)
        self.velocity_ub = to_torch(_env_cfg.velocity_ub, device=self._device)

        self._desired_velocity = torch.zeros(self._num_envs, 3, device=self._device)

        # ── Sim setup ─────────────────────────────────────────────────────────
        use_gpu = "cuda" in device
        self._sim_conf = sim_config.get_config(
            use_gpu=use_gpu,
            show_gui=show_gui,
            use_penetrating_contact=use_penetrating_contact,
        )
        self._gym, self._sim, self._viewer = create_sim(self._sim_conf)
        self._create_terrain()
        self._init_positions = self._compute_init_positions()

        # ── Robot ─────────────────────────────────────────────────────────────
        if use_real_robot:
            logger.warning(
                "[jump_env] use_real_robot=True — UnitreeQuad sim-only class used. "
                "Real robot out of scope."
            )

        logger.info("[jump_env] Creating UnitreeQuad robot")
        self._robot = unitree_quad.UnitreeQuad(
            num_envs           = self._num_envs,
            init_positions     = self._init_positions,
            sim                = self._sim,
            viewer             = self._viewer,
            sim_config         = self._sim_conf,
            motor_control_mode = MotorControlMode.HYBRID,
            robot_config       = self._config.robot,
        )

        if isinstance(strength_ratios, Sequence) and len(strength_ratios) == 2:
            ratios = torch_rand_float(
                lower=to_torch([strength_ratios[0]], device=self._device),
                upper=to_torch([strength_ratios[1]], device=self._device),
                shape=(self._num_envs, 3),
                device=self._device,
            )
            ratios = torch.concatenate((ratios, ratios, ratios, ratios), dim=1)
            self._robot.motor_group.strength_ratios = ratios
        else:
            self._robot.motor_group.strength_ratios = strength_ratios

        # Frictions set twice — Isaac Gym GPU requirement
        self._robot.set_foot_frictions(0.01)
        self._robot.set_foot_frictions(foot_friction)

        # ── Controllers ───────────────────────────────────────────────────────
        self._gait_generator = phase_gait_generator.PhaseGaitGenerator(
            self._robot, self._config.gait
        )
        self._swing_leg_controller = raibert_swing_leg_controller.RaibertSwingLegController(
            self._robot,
            self._gait_generator,
            raibert_config=self._config.raibert,
        )
        self._torque_optimizer = qp_torque_optimizer.QPTorqueOptimizer(
            self._robot,
            qp_config=self._config.qp,
        )

        self._steps_count    = torch.zeros(self._num_envs, device=self._device)
        self._init_yaw       = torch.zeros(self._num_envs, device=self._device)
        _env_cfg = self._config.env
        self._env_dt         = _env_cfg.env_dt
        self._episode_length = _env_cfg.episode_length_s / self._env_dt
        self._construct_observation_and_action_space()
        if observe_heights:
            self._height_points, self._num_height_points = self._compute_height_points()
        self._obs_buf             = None
        self._privileged_obs_buf  = None
        self._desired_landing_position = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float
        )
        self._cycle_count      = torch.zeros(self._num_envs, device=self._device)
        self._jumping_distance = torch.zeros((self._num_envs, 2), device=self._device)
        self._resample_command(torch.arange(self._num_envs, device=self._device))

        self._rewards = go1_rewards.Go1Rewards(self)
        self._prepare_rewards()
        self._extras = dict()

    def _create_terrain(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal           = gymapi.Vec3(0., 0., 1.)
        plane_params.static_friction  = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution      = 0.
        self._gym.add_ground(self._sim, plane_params)
        self._terrain = None

    def _compute_init_positions(self):
        init_height = self._init_height
        init_positions = torch.zeros((self._num_envs, 3), device=self._device)
        num_cols = int(np.sqrt(self._num_envs))
        distance = 1.
        for idx in range(self._num_envs):
            init_positions[idx, 0] = idx // num_cols * distance
            init_positions[idx, 1] = idx % num_cols * distance
            init_positions[idx, 2] = init_height
        return to_torch(init_positions, device=self._device)

    def _construct_observation_and_action_space(self):
        # Observation bounds from config.obs (Step 4 migration)
        _ow: dict = {}
        obs_cfg = self._config.obs

        robot_lb = to_torch(cfg(obs_cfg, "observation_robot_lb",
            [0.,-3.14,-3.14,-4.,-4.,-10.,-3.14,-3.14,-3.14]+[-0.5,-0.5,-0.4]*4,
            "config.obs", _ow), device=self._device)
        robot_ub = to_torch(cfg(obs_cfg, "observation_robot_ub",
            [0.6,3.14,3.14,8.,4.,10.,3.14,3.14,3.14]+[0.5,0.5,0.]*4,
            "config.obs", _ow), device=self._device)
        task_lb  = to_torch(cfg(obs_cfg, "task_lb",  [-2.,-2.,-1.,-1.,-1.], "config.obs", _ow),
                            device=self._device)
        task_ub  = to_torch(cfg(obs_cfg, "task_ub",  [ 2., 2., 1., 1., 1.], "config.obs", _ow),
                            device=self._device)

        flush_warnings(obs_cfg, "config.obs", _ow)

        vel_lb = torch.full((2,), self.velocity_lb, device=self._device)
        vel_ub = torch.full((2,), self.velocity_ub, device=self._device)
        self._observation_lb = torch.concatenate((task_lb, vel_lb, robot_lb))
        self._observation_ub = torch.concatenate((task_ub, vel_ub, robot_ub))

        if self._observe_heights:
            n = len(self._config.measured_points_x) * len(self._config.measured_points_y)
            h_lb = float(cfg(obs_cfg, "height_obs_lb", -3., "config.obs", {}))
            h_ub = float(cfg(obs_cfg, "height_obs_ub",  3., "config.obs", {}))
            self._observation_lb = torch.concatenate(
                (self._observation_lb, torch.zeros(n, device=self._device) + h_lb)
            )
            self._observation_ub = torch.concatenate(
                (self._observation_ub, torch.zeros(n, device=self._device) + h_ub)
            )

        if self._learn_raibert_alpha:
            base_action_lb = to_torch(self._config.env.action_lb, device=self._device)
            base_action_ub = to_torch(self._config.env.action_ub, device=self._device)
            alpha_lb = to_torch(cfg(obs_cfg, "alpha_lb", [0.5], "config.obs", {}),
                                device=self._device)
            alpha_ub = to_torch(cfg(obs_cfg, "alpha_ub", [2.0], "config.obs", {}),
                                device=self._device)
            self._action_lb = torch.concatenate([base_action_lb, alpha_lb])
            self._action_ub = torch.concatenate([base_action_ub, alpha_ub])
        else:
            self._action_lb = to_torch(self._config.env.action_lb, device=self._device)
            self._action_ub = to_torch(self._config.env.action_ub, device=self._device)

    def _prepare_rewards(self):
        self._reward_names, self._reward_fns, self._reward_scales = [], [], []
        self._episode_sums = dict()
        for name, scale in self._config.rewards.rewards:
            self._reward_names.append(name)
            self._reward_fns.append(getattr(self._rewards, name + "_reward"))
            self._reward_scales.append(scale)
            self._episode_sums[name] = torch.zeros(self._num_envs, device=self._device)
        self._terminal_reward_names, self._terminal_reward_fns, self._terminal_reward_scales = [], [], []
        for name, scale in self._config.rewards.terminal_rewards:
            self._terminal_reward_names.append(name)
            self._terminal_reward_fns.append(getattr(self._rewards, name + "_reward"))
            self._terminal_reward_scales.append(scale)
            self._episode_sums[name] = torch.zeros(self._num_envs, device=self._device)

    def reset(self):
        return self.reset_idx(torch.arange(self._num_envs, device=self._device))

    def _split_action(self, action):
        alpha = None
        if self._learn_raibert_alpha:
            alpha  = action[:, -1:]
            action = action[:, :-1]
        gait_action = None
        if self._include_gait_action:
            gait_action = action[:, :1]
            action      = action[:, 1:]
        foot_action = None
        if self._include_foot_action:
            if self._mirror_foot_action:
                foot_action = action[:, -6:].reshape((-1, 2, 3))
                foot_action = torch.stack([
                    foot_action[:, 0], foot_action[:, 0],
                    foot_action[:, 1], foot_action[:, 1],
                ], dim=1)
                action = action[:, :-6]
            else:
                foot_action = action[:, -12:].reshape((-1, 4, 3))
                action      = action[:, :-12]
        return gait_action, action, foot_action, alpha

    def reset_idx(self, env_ids):
        self._extras["time_outs"] = self._episode_terminated()
        if env_ids.shape[0] > 0:
            self._extras["episode"] = {}
            self._extras["episode"]["cycle_count"] = torch.mean(
                self._gait_generator.true_phase[env_ids]
            ) / (2 * torch.pi)

            self._obs_buf            = self._get_observations()
            self._privileged_obs_buf = self._get_privileged_observations()

            for reward_name in self._episode_sums.keys():
                if reward_name in self._reward_names:
                    if self._normalize_by_phase:
                        self._extras["episode"][f"reward_{reward_name}"] = torch.mean(
                            self._episode_sums[reward_name][env_ids]
                            / (self._gait_generator.true_phase[env_ids] / (2 * torch.pi))
                        )
                    else:
                        self._extras["episode"][f"reward_{reward_name}"] = torch.mean(
                            self._episode_sums[reward_name][env_ids]
                            / (self._steps_count[env_ids] * self._env_dt)
                        )
                if reward_name in self._terminal_reward_names:
                    self._extras["episode"][f"reward_{reward_name}"] = torch.mean(
                        self._episode_sums[reward_name][env_ids]
                        / (self._cycle_count[env_ids].clip(min=1))
                    )
                self._episode_sums[reward_name][env_ids] = 0

            r = torch.rand(env_ids.shape[0], device=self.device)
            desired_velocity_x = self.velocity_lb + r * (self.velocity_ub - self.velocity_lb)
            self._desired_velocity[env_ids, 0] = desired_velocity_x
            self._desired_velocity[env_ids, 1:] = 0

            self._steps_count[env_ids]  = 0
            self._cycle_count[env_ids]  = 0
            self._init_yaw[env_ids]     = self._robot.base_orientation_rpy[env_ids, 2]
            self._robot.reset_idx(env_ids)
            self._swing_leg_controller.reset_idx(env_ids)
            self._gait_generator.reset_idx(env_ids)
            self._resample_command(env_ids)

        return self._obs_buf, self._privileged_obs_buf

    def step(self, action: torch.Tensor):
        self._last_action = torch.clone(action)
        action = torch.clip(action, self._action_lb, self._action_ub)
        sum_reward = torch.zeros(self._num_envs, device=self._device)
        dones      = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        self._steps_count += 1
        logs = []
        zero = torch.zeros(self._num_envs, device=self._device)

        gait_action, com_action, foot_action, alpha = self._split_action(action)

        if alpha is not None:
            self._swing_leg_controller.set_kp_alpha(alpha)

        desired_linear_vel_z = (
            com_action[:, 2] - self._torque_optimizer.desired_base_position[:, 2]
        ) / self._env_dt
        desired_linear_vel_z = desired_linear_vel_z.clip(min=-0., max=0.)

        desired_ang_vel_y = (
            com_action[:, 4] - self._torque_optimizer.desired_base_orientation_rpy[:, 1]
        ) / self._env_dt
        desired_ang_vel_y = desired_ang_vel_y.clip(min=-0., max=0.)

        for step in range(max(int(self._env_dt / self._robot.control_timestep), 1)):
            self._gait_generator.update()
            self._swing_leg_controller.update(self._desired_velocity)

            if gait_action is not None:
                self._gait_generator.stepping_frequency = gait_action[:, 0]

            self._torque_optimizer.desired_base_position = torch.stack(
                (self._robot.base_position[:, 0],
                 self._robot.base_position[:, 1],
                 com_action[:, 0]), dim=1
            )
            self._torque_optimizer.desired_linear_velocity = torch.stack(
                (com_action[:, 1], com_action[:, 2] * 0, com_action[:, 3]), dim=1
            )
            self._torque_optimizer.desired_base_orientation_rpy = torch.stack(
                (com_action[:, 4] * 0,
                 com_action[:, 5],
                 self._robot.base_orientation_rpy[:, 2]), dim=1
            )

            if self._use_yaw_feedback:
                yaw_err = self._init_yaw - self._robot.base_orientation_rpy[:, 2]
                yaw_err = torch.remainder(yaw_err + 3 * torch.pi, 2 * torch.pi) - torch.pi
                desired_yaw_rate = self._yaw_feedback_gain * yaw_err
                self._torque_optimizer.desired_angular_velocity = torch.stack(
                    (zero, com_action[:, 6], desired_yaw_rate), dim=1
                )
            else:
                self._torque_optimizer.desired_angular_velocity = torch.stack(
                    (zero, com_action[:, 6], com_action[:, 7] * 0), dim=1
                )

            desired_foot_positions = self._swing_leg_controller.desired_foot_positions
            if foot_action is not None:
                base_yaw = self._robot.base_orientation_rpy[:, 2]
                cos_yaw  = torch.cos(base_yaw)[:, None]
                sin_yaw  = torch.sin(base_yaw)[:, None]
                foot_action_world = torch.clone(foot_action)
                foot_action_world[:, :, 0] = (
                    cos_yaw * foot_action[:, :, 0] - sin_yaw * foot_action[:, :, 1]
                )
                foot_action_world[:, :, 1] = (
                    sin_yaw * foot_action[:, :, 0] + cos_yaw * foot_action[:, :, 1]
                )
                desired_foot_positions += foot_action_world

            motor_action, self._desired_acc, self._solved_acc, self._qp_cost, self._num_clips = \
                self._torque_optimizer.get_action(
                    self._gait_generator.desired_contact_state,
                    swing_foot_position=desired_foot_positions,
                )

            logs.append(dict(
                timestamp=self._robot.time_since_reset,
                base_position=torch.clone(self._robot.base_position),
                base_orientation_rpy=torch.clone(self._robot.base_orientation_rpy),
                base_velocity=torch.clone(self._robot.base_velocity_body_frame),
                base_angular_velocity=torch.clone(self._robot.base_angular_velocity_body_frame),
                motor_positions=torch.clone(self._robot.motor_positions),
                motor_velocities=torch.clone(self._robot.motor_velocities),
                motor_action=motor_action,
                motor_torques=self._robot.motor_torques,
                num_clips=self._num_clips,
                foot_contact_state=self._gait_generator.desired_contact_state,
                foot_contact_force=self._robot.foot_contact_forces,
                desired_swing_foot_position=desired_foot_positions,
                desired_acc_body_frame=self._desired_acc,
                solved_acc_body_frame=self._solved_acc,
                foot_positions_in_base_frame=self._robot.foot_positions_in_base_frame,
                env_action=action,
                env_obs=torch.clone(self._obs_buf),
            ))

            self._robot.step(motor_action)
            self._obs_buf            = self._get_observations()
            self._privileged_obs_buf = self.get_privileged_observations()
            rewards = self.get_reward()
            dones   = torch.logical_or(dones, self._is_done())
            sum_reward += rewards * torch.logical_not(dones)

        if self._show_gui and self._frame_callback is not None:
            if self._frame_callback is not None: 
                self._frame_callback(self._robot.time_since_reset.item()) # This  draws overlay everyframe!
            self._robot.render()
            if self._capture_callback is not None: 
                self._capture_callback() # This is called every timestep, but does not capture every time its called 


        self._extras["logs"] = logs

        new_cycle_count = (self._gait_generator.true_phase / (2 * torch.pi)).long()
        finished_cycle  = new_cycle_count > self._cycle_count
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
        init_height = self._init_height
        if self._num_envs == 1 and self._jumping_distance_schedule is not None:
            self._jumping_distance = torch.tensor(
                [[next(self._jumping_distance_schedule), 0]], device=self._device
            )
        else:
            self._jumping_distance[env_ids] = torch_rand_float(
                self._config.env.goal_lb, self._config.env.goal_ub,
                [env_ids.shape[0], 2], device=self._device,
            )
        self._desired_landing_position[env_ids, :2] = (
            self._robot.base_position[env_ids, :2]
            + gravity_frame_to_world_frame(
                self._robot.base_orientation_rpy[env_ids, 2],
                self._jumping_distance[env_ids],
            )
        )
        self._desired_landing_position[env_ids, 2] = init_height

    def _get_observations(self):
        distance_to_goal = self._desired_landing_position - self._robot.base_position_world
        distance_to_goal_local = world_frame_to_gravity_frame(
            self._robot.base_orientation_rpy[:, 2], distance_to_goal
        )
        phase_obs = torch.stack((
            torch.cos(self._gait_generator.true_phase),
            torch.sin(self._gait_generator.true_phase),
        ), dim=1)
        velocity_command = self._desired_velocity[:, :2]
        robot_obs = torch.concatenate((
            self._robot.base_position[:, 2:],
            self._robot.base_orientation_rpy[:, 0:1] * 0,
            self._robot.base_orientation_rpy[:, 1:2],
            self._robot.base_velocity_body_frame[:, 0:1],
            self._robot.base_velocity_body_frame[:, 1:2] * 0,
            self._robot.base_velocity_body_frame[:, 2:3],
            self._robot.base_angular_velocity_body_frame[:, 0:1] * 0,
            self._robot.base_angular_velocity_body_frame[:, 1:2],
            self._robot.base_angular_velocity_body_frame[:, 2:3],
            self._robot.foot_positions_in_base_frame.reshape((self._num_envs, 12)),
        ), dim=1)
        obs = torch.concatenate(
            (distance_to_goal_local, phase_obs, velocity_command, robot_obs), dim=1
        )
        if self._has_obs_noise and not self._use_real_robot:
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
            reward_item = self._reward_scales[idx] * self._reward_fns[idx]()
            if self._normalize_by_phase:
                reward_item *= self._gait_generator.stepping_frequency
            self._episode_sums[self._reward_names[idx]] += reward_item
            sum_reward += reward_item
        if self._config.rewards.clip_negative_reward:
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

        if self._config.rewards.clip_negative_terminal_reward:
            sum_reward = torch.clip(sum_reward, min=0)
        return sum_reward

    def _episode_terminated(self):
        timeout = self._steps_count >= self._episode_length
        cycles_finished = (
            self._gait_generator.true_phase / (2 * torch.pi)
        ) > self._max_jumps
        return torch.logical_or(timeout, cycles_finished)

    def _is_done(self):
        is_unsafe = torch.logical_or(
            self._robot.projected_gravity[:, 2] < self._terminate_on_gravity_z,
            self._robot.base_position[:, 2] < self._terminate_on_height,
        )
        if self._terminate_on_body:
            is_unsafe = torch.logical_or(is_unsafe, self._robot.has_body_contact)
        if self._terminate_on_limb:
            limb_contact = torch.logical_or(
                self._robot.calf_contacts, self._robot.thigh_contacts
            )
            is_unsafe = torch.logical_or(is_unsafe, torch.sum(limb_contact, dim=1) > 0)
        return torch.logical_or(self._episode_terminated(), is_unsafe)

    # ── Properties ────────────────────────────────────────────────────────────
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
            new_length / self.max_episode_length
            * self._max_jumps + 1
        )[:, None]
        self._cycle_count = (self._gait_generator.true_phase / (2 * torch.pi)).long()
