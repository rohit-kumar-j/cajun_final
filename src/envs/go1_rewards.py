"""<src/envs/go1_rewards.py>Set of rewards for the Go1/Go2 robot.

Step 5 changes:
  - foot_radius read from config.robot.foot_radius (was 0.02, URDF = 0.022)
  - foot_clearance_threshold read from config.rewards via cfg()
  - motor_heat_coefficient read from config.rewards via cfg()
  - energy_clip_max read from config.rewards via cfg()
  - com_height_reward_max read from config.rewards via cfg()
  - height_reward target uses config.robot.init_height (fixes 0.26 vs 0.268 bug)
  - All reward functions unchanged — only threshold/coefficient reads updated
"""
from loguru import logger
import torch

from src.utilities.config_utils import cfg, flush_warnings, REQUIRED


class Go1Rewards:
    """Reward functions for the Go2 robot."""

    def __init__(self, env):
        self._env           = env
        self._robot         = self._env.robot
        self._gait_generator = self._env.gait_generator
        self._num_envs      = self._env.num_envs
        self._device        = self._env.device

        # Read reward-specific config values once at init via cfg()
        rewards_cfg = self._env._config.rewards
        robot_cfg   = self._env._config.robot

        _w: dict = {}
        C = "Go1Rewards"

        self._foot_clearance_threshold = float(cfg(rewards_cfg, "foot_clearance_threshold", REQUIRED, C, _w))
        self._motor_heat_coefficient   = float(cfg(rewards_cfg, "motor_heat_coefficient",   REQUIRED, C, _w))
        self._energy_clip_max          = float(cfg(rewards_cfg, "energy_clip_max",          REQUIRED, C, _w))
        self._com_height_reward_max    = float(cfg(rewards_cfg, "com_height_reward_max",    REQUIRED, C, _w))
        self._limb_force_clip_max      = float(cfg(rewards_cfg, "limb_force_clip_max",      REQUIRED, C, _w))
        self._min_stepping_freq        = float(cfg(rewards_cfg, "min_stepping_freq_reward", REQUIRED, C, _w))

        # foot_radius from config.robot — URDF sphere radius = 0.022 m (was 0.02 hardcoded)
        _rw: dict = {}
        self._foot_radius    = float(cfg(robot_cfg, "foot_radius",  0.022, C, _rw))
        # height_reward target — unified with config.robot.init_height
        self._target_height  = float(cfg(robot_cfg, "init_height",  0.268, C, _rw))

        flush_warnings(rewards_cfg, C, _w)
        flush_warnings(robot_cfg,   C, _rw)

        logger.info(
            f"[rewards] foot_radius={self._foot_radius}  "
            f"clearance_thresh={self._foot_clearance_threshold}  "
            f"heat_coeff={self._motor_heat_coefficient}  "
            f"height_target={self._target_height}"
        )

    # ── Reward functions ───────────────────────────────────────────────────────

    def speed_tracking_reward(self):
        actual = torch.cat([
            self._robot.base_velocity_body_frame[:, :2],
            self._robot.base_angular_velocity_body_frame[:, 2:3],
        ], dim=1)
        desired = torch.cat([
            self._env._desired_velocity[:, :2],
            torch.zeros(self._num_envs, 1, device=self._device),
        ], dim=1)
        return -torch.sum(torch.square(desired - actual), dim=1)

    def forward_speed_reward(self):
        return self._robot.base_velocity_body_frame[:, 0]

    def upright_reward(self):
        return self._robot.projected_gravity[:, 2]

    def alive_reward(self):
        return torch.ones(self._num_envs, device=self._device)

    def height_reward(self):
        return -torch.square(self._robot.base_position[:, 2] - self._target_height)

    def foot_slipping_reward(self):
        foot_slipping = torch.sum(
            self._gait_generator.desired_contact_state
            * torch.sum(torch.square(
                self._robot.foot_velocities_in_world_frame[:, :, :2]
            ), dim=2),
            dim=1,
        ) / 4
        return -torch.clip(foot_slipping, 0, 1)

    def foot_clearance_reward(self):
        desired_contacts = self._gait_generator.desired_contact_state
        foot_height = self._robot.foot_height - self._foot_radius
        foot_height = torch.clip(foot_height, 0, self._foot_clearance_threshold) / self._foot_clearance_threshold
        foot_clearance = torch.sum(
            torch.logical_not(desired_contacts) * foot_height, dim=1
        ) / 4
        return foot_clearance

    def foot_force_reward(self):
        foot_forces  = torch.norm(self._robot.foot_contact_forces,  dim=2)
        calf_forces  = torch.norm(self._robot.calf_contact_forces,  dim=2)
        thigh_forces = torch.norm(self._robot.thigh_contact_forces, dim=2)
        limb_forces  = (foot_forces + calf_forces + thigh_forces).clip(max=self._limb_force_clip_max)
        foot_mask    = torch.logical_not(self._gait_generator.desired_contact_state)
        return -torch.sum(limb_forces * foot_mask, dim=1) / 4

    def cost_of_transport_reward(self):
        motor_power  = torch.abs(
            self._motor_heat_coefficient * self._robot.motor_torques**2
            + self._robot.motor_torques * self._robot.motor_velocities
        )
        commanded_vel = torch.sqrt(
            torch.sum(self._env._desired_velocity[:, :2]**2, dim=1)
        )
        return -torch.sum(motor_power, dim=1) / commanded_vel

    def energy_consumption_reward(self):
        motor_power = torch.clip(
            self._motor_heat_coefficient * self._robot.motor_torques**2
            + self._robot.motor_torques * self._robot.motor_velocities,
            min=0,
        )
        return -torch.clip(
            torch.sum(motor_power, dim=1), min=0., max=self._energy_clip_max
        )

    def contact_consistency_reward(self):
        desired_contact = self._gait_generator.desired_contact_state
        actual_contact  = torch.logical_or(
            self._robot.foot_contacts, self._robot.calf_contacts
        )
        actual_contact  = torch.logical_or(actual_contact, self._robot.thigh_contacts)
        return torch.sum(desired_contact == actual_contact, dim=1) / 4

    def distance_to_goal_reward(self):
        base_position = self._robot.base_position_world
        return -torch.sqrt(torch.sum(torch.square(
            base_position[:, :2] - self._env.desired_landing_position[:, :2]
        ), dim=1))

    def com_distance_to_goal_squared_reward(self):
        base_position = self._robot.base_position_world
        return -torch.sum(torch.square(
            (base_position[:, :2] - self._env.desired_landing_position[:, :2])
            / (self._env._jumping_distance[:, 0:1])
        ), dim=1)

    def swing_foot_vel_reward(self):
        foot_vel     = torch.sum(self._robot.foot_velocities_in_base_frame**2, dim=2)
        contact_mask = torch.logical_not(self._env.gait_generator.desired_contact_state)
        return -torch.sum(foot_vel * contact_mask, dim=1) / (
            torch.sum(contact_mask, dim=1) + 0.001
        )

    def com_height_reward(self):
        return self._robot.base_position_world[:, 2].clip(max=self._com_height_reward_max)

    def heading_reward(self):
        return -self._robot.base_orientation_rpy[:, 2]**2

    def out_of_bound_action_reward(self):
        exceeded = torch.maximum(
            self._env._action_lb - self._env._last_action,
            self._env._last_action - self._env._action_ub,
        )
        exceeded = torch.clip(exceeded, min=0.)
        normalized = exceeded / (self._env._action_ub - self._env._action_lb)
        return -torch.sum(torch.square(normalized), dim=1)

    def swing_residual_reward(self):
        return -torch.mean(torch.square(self._env._last_action[:, -6:]), axis=1)

    def knee_contact_reward(self):
        return -(torch.sum(
            torch.logical_or(self._env.robot.thigh_contacts, self._env.robot.calf_contacts),
            dim=1,
        ).float()) / 4

    def body_contact_reward(self):
        return -self._robot.has_body_contact.float()

    def stepping_freq_reward(self):
        return self._min_stepping_freq - self._env.gait_generator.stepping_frequency.clip(min=self._min_stepping_freq)

    def friction_cone_reward(self):
        return -self._env._num_clips / 4
