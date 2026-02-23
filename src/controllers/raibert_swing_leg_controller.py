"""<src/controllers/raibert_swing_leg_controller.py>
Raibert Swing Leg controller.

Step 3 changes:
  - All magic values removed from __init__ defaults
  - Values read from config.raibert via cfg() in __init__
  - Velocity-dependent kp coefficients read from config
  - loguru added
"""
from typing import Any

from isaacgym.torch_utils import quat_rotate_inverse, to_torch
from loguru import logger
import torch

from src.controllers import phase_gait_generator
from src.utilities.config_utils import cfg, flush_warnings, REQUIRED


@torch.jit.script
def cubic_bezier(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    progress = t**3 + 3 * t**2 * (1 - t)
    return x0 + progress * (x1 - x0)


@torch.jit.script
def _gen_swing_foot_trajectory(
    input_phase: torch.Tensor,
    start_pos:   torch.Tensor,
    mid_pos:     torch.Tensor,
    end_pos:     torch.Tensor,
) -> torch.Tensor:
    cutoff     = 0.5
    input_phase = torch.stack([input_phase] * 3, dim=-1)
    return torch.where(
        input_phase < cutoff,
        cubic_bezier(start_pos, mid_pos, input_phase / cutoff),
        cubic_bezier(mid_pos, end_pos, (input_phase - cutoff) / (1 - cutoff)),
    )


@torch.jit.script
def cross_quad(v1, v2):
    """v1 is nx3, v2 is nx4x3"""
    v1 = torch.stack([v1, v1, v1, v1], dim=1)
    shape = v1.shape
    v1 = v1.reshape((-1, 3))
    v2 = v2.reshape((-1, 3))
    return torch.cross(v1, v2).reshape((shape[0], shape[1], 3))


@torch.jit.script
def compute_desired_foot_positions(
    base_rot_mat,
    base_height,
    hip_positions_in_body_frame,
    base_velocity_world_frame,
    base_angular_velocity_body_frame,
    projected_gravity,
    desired_base_height:   float,
    foot_height:           float,
    foot_landing_clearance: float,
    stance_duration,
    normalized_phase,
    phase_switch_foot_positions,
    current_vel,
    desired_vel,
    raibert_kp,
):
    hip_position = torch.matmul(
        base_rot_mat, hip_positions_in_body_frame.transpose(1, 2)
    ).transpose(1, 2)
    mid_position = torch.clone(hip_position)
    mid_position[..., 2] = -base_height[:, None] + foot_height

    base_velocity = base_velocity_world_frame
    hip_velocity_body_frame = cross_quad(
        base_angular_velocity_body_frame, hip_positions_in_body_frame
    )
    current_hip_velocity = base_velocity[:, None, :] + torch.matmul(
        base_rot_mat, hip_velocity_body_frame.transpose(1, 2)
    ).transpose(1, 2)

    desired_angular_vel = torch.zeros_like(base_angular_velocity_body_frame)
    desired_hip_velocity_body_frame = cross_quad(
        desired_angular_vel, hip_positions_in_body_frame
    )
    desired_hip_velocity = desired_vel[:, None, :] + torch.matmul(
        base_rot_mat, desired_hip_velocity_body_frame.transpose(1, 2)
    ).transpose(1, 2)

    land_position = (
        current_hip_velocity * stance_duration[:, :, None] / 2
        + raibert_kp * (current_hip_velocity - desired_hip_velocity)
    )
    land_position += hip_position
    land_position[..., 2] = -base_height[:, None] + foot_landing_clearance

    foot_position = _gen_swing_foot_trajectory(
                    normalized_phase,
                    phase_switch_foot_positions,
                    mid_position,
                    land_position,
    )
    return foot_position


class RaibertSwingLegController:
    """Controls swing leg position using Raibert's heuristic.

    All parameters come from config.raibert — no defaults in __init__.
    """

    def __init__(
        self,
        robot:          Any,
        gait_generator: Any,
        raibert_config: Any,
    ):
        self._robot          = robot
        self._device         = self._robot._device
        self._num_envs       = self._robot.num_envs
        self._gait_generator = gait_generator
        self._last_leg_state = gait_generator.desired_contact_state

        _w: dict = {}
        C = "RaibertSwingLegController"

        self._desired_base_height    = float(cfg(raibert_config, "desired_base_height",    REQUIRED, C, _w))
        self._foot_height            = float(cfg(raibert_config, "foot_height",            REQUIRED, C, _w))
        self._foot_landing_clearance = float(cfg(raibert_config, "foot_landing_clearance", REQUIRED, C, _w))
        self._default_raibert_kp     = float(cfg(raibert_config, "default_kp",            REQUIRED, C, _w))
        self._kp_base                = float(cfg(raibert_config, "kp_base",               REQUIRED, C, _w))
        self._kp_vel_scale           = float(cfg(raibert_config, "kp_vel_scale",          REQUIRED, C, _w))
        self._kp_min                 = float(cfg(raibert_config, "kp_min",                REQUIRED, C, _w))
        self._kp_max                 = float(cfg(raibert_config, "kp_max",                REQUIRED, C, _w))

        flush_warnings(raibert_config, C, _w)

        logger.info(
            f"[raibert] foot_height={self._foot_height}  "
            f"landing_clearance={self._foot_landing_clearance}  "
            f"kp_base={self._kp_base}  kp_range=[{self._kp_min},{self._kp_max}]"
        )

        self._raibert_kp = torch.full(
            (self._num_envs, 1, 1), self._default_raibert_kp, device=self._device
        )
        self._alpha = torch.ones((self._num_envs, 1), device=self._device)
        self._phase_switch_foot_positions = None
        self.reset()

    def set_kp_alpha(self, alpha: torch.Tensor) -> None:
        self._alpha = alpha

    def reset(self) -> None:
        self._last_leg_state = torch.clone(self._gait_generator.desired_contact_state)
        self._phase_switch_foot_positions = torch.matmul(
            self._robot.base_rot_mat,
            self._robot.foot_positions_in_base_frame.transpose(1, 2)).transpose(1, 2)

    def reset_idx(self, env_ids) -> None:
        self._last_leg_state[env_ids] = torch.clone(
            self._gait_generator.desired_contact_state[env_ids]
        )
        self._phase_switch_foot_positions[env_ids] = torch.matmul(
            self._robot.base_rot_mat[env_ids],
            self._robot.foot_positions_in_base_frame[env_ids].transpose(1, 2)).transpose(1, 2)

    def update(self, desired_velocity) -> None:
        self._new_desired_velocity = desired_velocity
        current_velocity_mag = torch.norm(
            self._robot.base_velocity_world_frame, dim=-1, keepdim=True
        ).clamp(min=1e-6)

        base_kp = self._kp_base + self._kp_vel_scale * current_velocity_mag.clamp(max=5.0)
        self._raibert_kp = (base_kp * self._alpha).clamp(
            min=self._kp_min, max=self._kp_max
        ).unsqueeze(-1)

        new_leg_state     = torch.clone(self._gait_generator.desired_contact_state)
        new_foot_positions = torch.matmul(
            self._robot.base_rot_mat,
            self._robot.foot_positions_in_base_frame.transpose(1, 2)).transpose(1, 2)
        self._phase_switch_foot_positions = torch.where(
            torch.tile((self._last_leg_state == new_leg_state)[:, :, None], [1, 1, 3]),
            self._phase_switch_foot_positions,
            new_foot_positions
        )
        self._last_leg_state = new_leg_state

    @property
    def desired_foot_positions(self):
        return compute_desired_foot_positions(
            self._robot.base_rot_mat,
            self._robot.base_position[:, 2],
            self._robot.hip_positions_in_body_frame,
            self._robot.base_velocity_world_frame,
            self._robot.base_angular_velocity_body_frame,
            self._robot.projected_gravity,
            self._desired_base_height,
            self._foot_height,
            self._foot_landing_clearance,
            self._gait_generator.stance_duration,
            self._gait_generator.normalized_phase,
            self._phase_switch_foot_positions,
            self._robot.base_vel,
            self._new_desired_velocity,
            self._raibert_kp,
        )
