"""<src/controllers/qp_torque_optimizer.py>
Centroidal QP controller to compute desired foot forces and joint torques.

Step 3 changes:
  - All magic value defaults removed from __init__ signature
  - Values read from config.qp via cfg() in __init__
  - stance kp/kd (were hardcoded 60/2 in compute_joint_command) now from config.qp
  - acceleration bounds (acc_lb/ub) now from config.qp
  - foot workspace height limits (foot_height_lb/ub) now from config.qp
  - GRF z bounds (grf_z_lb/ub) now from config.qp
  - body_mass now from config.qp (was 13.076 = Go1 default)
  - get_action_with_acc removed (was unused)
  - loguru added
"""
import time

from isaacgym.torch_utils import quat_mul, quat_from_euler_xyz, to_torch, quat_rotate
from loguru import logger
import numpy as np
import torch
from qpth.qp import QPFunction, QPSolvers

from src.robots.motors import MotorCommand
from src.utilities.config_utils import cfg, flush_warnings, REQUIRED
from src.utilities.rotation_utils import quat_to_rot_mat


@torch.jit.script
def quaternion_to_axis_angle(q):
    angle = 2 * torch.acos(torch.clip(q[:, 3], -0.99999, 0.99999))[:, None]
    norm  = torch.clip(torch.linalg.norm(q[:, :3], dim=1), 1e-5, 1)[:, None]
    axis  = q[:, :3] / norm
    return axis, angle


@torch.jit.script
def compute_orientation_error(
    desired_orientation_rpy, base_orientation_quat, device: str = 'cuda'
):
    desired_quat = quat_from_euler_xyz(
        desired_orientation_rpy[:, 0], desired_orientation_rpy[:, 1],
        torch.zeros_like(desired_orientation_rpy[:, 2])
    )
    base_quat_inv = torch.clone(base_orientation_quat)
    base_quat_inv[:, -1] *= -1
    error_quat = quat_mul(desired_quat, base_quat_inv)
    axis, angle = quaternion_to_axis_angle(error_quat)
    angle = torch.where(angle > torch.pi, angle - 2 * torch.pi, angle)
    return quat_rotate(base_orientation_quat, axis * angle)


@torch.jit.script
def compute_desired_acc(
    base_orientation_rpy, base_position, base_angular_velocity_body_frame,
    base_velocity_body_frame, desired_base_orientation_rpy, desired_base_position,
    desired_angular_velocity, desired_linear_velocity,
    desired_angular_acceleration, desired_linear_acceleration,
    base_position_kp, base_position_kd, base_orientation_kp, base_orientation_kd,
    device: str = "cuda",
):
    base_rpy  = base_orientation_rpy
    base_quat = quat_from_euler_xyz(
        base_rpy[:, 0], base_rpy[:, 1],
        torch.zeros_like(base_rpy[:, 0], device=device)
    )
    base_rot_mat   = quat_to_rot_mat(base_quat)
    base_rot_mat_t = torch.transpose(base_rot_mat, 1, 2)

    lin_pos_error = desired_base_position - base_position
    lin_pos_error[:, :2] = 0
    lin_vel_error = desired_linear_velocity - torch.matmul(
        base_rot_mat, base_velocity_body_frame[:, :, None]
    )[:, :, 0]
    desired_lin_acc_gf = (
          base_position_kp * lin_pos_error
        + base_position_kd * lin_vel_error
        + desired_linear_acceleration
    )

    ang_pos_error = compute_orientation_error(
        desired_base_orientation_rpy, base_quat, device=device
    )
    ang_vel_error = desired_angular_velocity - torch.matmul(
        base_rot_mat, base_angular_velocity_body_frame[:, :, None]
    )[:, :, 0]
    desired_ang_acc_gf = (
          base_orientation_kp * ang_pos_error
        + base_orientation_kd * ang_vel_error
        + desired_angular_acceleration
    )

    desired_lin_acc_body = torch.matmul(
        base_rot_mat_t, desired_lin_acc_gf[:, :, None]
    )[:, :, 0]
    desired_ang_acc_body = torch.matmul(
        base_rot_mat_t, desired_ang_acc_gf[:, :, None]
    )[:, :, 0]
    return torch.concatenate((desired_lin_acc_body, desired_ang_acc_body), dim=1)


@torch.jit.script
def convert_to_skew_symmetric_batch(foot_positions):
    n    = foot_positions.shape[0]
    x, y, z = foot_positions[:,:,0], foot_positions[:,:,1], foot_positions[:,:,2]
    zero = torch.zeros_like(x)
    skew = torch.stack([zero,-z,y, z,zero,-x, -y,x,zero], dim=1).reshape((n,3,3,4))
    return torch.concatenate(
        [skew[:,:,:,0], skew[:,:,:,1], skew[:,:,:,2], skew[:,:,:,3]], dim=2
    )


@torch.jit.script
def construct_mass_mat(
    foot_positions, foot_contact_state, inv_mass, inv_inertia,
    device: str = 'cuda', mask_noncontact_legs: bool = True
):
    num_envs = foot_positions.shape[0]
    mass_mat = torch.zeros((num_envs, 6, 12), device=device)
    inv_mass_concat = torch.concatenate([inv_mass] * 4, dim=1)
    mass_mat[:, :3] = inv_mass_concat[None, :, :]
    px = convert_to_skew_symmetric_batch(foot_positions)
    mass_mat[:, 3:6] = torch.matmul(inv_inertia, px)
    if mask_noncontact_legs:
        non_contact = torch.nonzero(torch.logical_not(foot_contact_state))
        env_id, leg_id = non_contact[:, 0], non_contact[:, 1]
        mass_mat[env_id, :, leg_id*3]   = 0
        mass_mat[env_id, :, leg_id*3+1] = 0
        mass_mat[env_id, :, leg_id*3+2] = 0
    return mass_mat


@torch.jit.script
def solve_grf(
    mass_mat, desired_acc, base_rot_mat_t, Wq, Wf: float,
    foot_friction_coef: float, clip_grf: bool, foot_contact_state,
    grf_z_lb: float, grf_z_ub: float,
    device: str = 'cuda',
):
    num_envs = mass_mat.shape[0]
    g = torch.zeros((num_envs, 6), device=device)
    g[:, 2] = 9.8
    g[:, :3] = torch.matmul(base_rot_mat_t, g[:, :3, None])[:, :, 0]

    Q       = torch.zeros((num_envs, 6, 6), device=device) + Wq[None, :]
    Wf_mat  = torch.eye(12, device=device) * Wf
    R       = torch.zeros((num_envs, 12, 12), device=device) + Wf_mat[None, :]

    quad_term   = torch.bmm(torch.bmm(torch.transpose(mass_mat,1,2), Q), mass_mat) + R
    linear_term = torch.bmm(
        torch.bmm(torch.transpose(mass_mat,1,2), Q),
        (g + desired_acc)[:, :, None]
    )[:, :, 0]
    grf = torch.linalg.solve(quad_term, linear_term)

    base_rot_mat = torch.transpose(base_rot_mat_t, 1, 2)
    grf          = grf.reshape((-1, 4, 3))
    grf_world    = torch.transpose(
        torch.bmm(base_rot_mat, torch.transpose(grf, 1, 2)), 1, 2
    )
    if clip_grf:
        grf_world[:, :, 2] = grf_world[:, :, 2].clip(min=grf_z_lb, max=grf_z_ub)
        grf_world[:, :, 2] *= foot_contact_state

    friction_force    = torch.norm(grf_world[:, :, :2], dim=2) + 0.001
    max_friction      = foot_friction_coef * grf_world[:, :, 2].clip(min=0)
    multiplier        = torch.where(
        friction_force < max_friction, 1, max_friction / friction_force
    )
    if clip_grf:
        grf_world[:, :, :2] *= multiplier[:, :, None]
    grf = torch.transpose(
        torch.bmm(base_rot_mat_t, torch.transpose(grf_world, 1, 2)), 1, 2
    ).reshape((-1, 12))

    # Convert to motor torques
    solved_acc = torch.bmm(mass_mat, grf[:, :, None])[:, :, 0] - g
    qp_cost    = torch.bmm(
        torch.bmm((solved_acc - desired_acc)[:,:,None].transpose(1,2), Q),
        (solved_acc - desired_acc)[:,:,None]
    )[:, 0, 0]
    return grf, solved_acc, qp_cost, torch.sum(friction_force > max_friction + 1, dim=1)


class QPTorqueOptimizer:
    """Centroidal QP controller. All parameters from config.qp — no defaults."""

    def __init__(self, robot, qp_config):
        self._robot    = robot
        self._device   = self._robot._device
        self._num_envs = self._robot.num_envs

        _w: dict = {}
        C = "QPTorqueOptimizer"

        base_position_kp    = cfg(qp_config, "base_position_kp",    REQUIRED, C, _w)
        base_position_kd    = cfg(qp_config, "base_position_kd",    REQUIRED, C, _w)
        base_orientation_kp = cfg(qp_config, "base_orientation_kp", REQUIRED, C, _w)
        base_orientation_kd = cfg(qp_config, "base_orientation_kd", REQUIRED, C, _w)
        weight_ddq          = cfg(qp_config, "weight_ddq",          REQUIRED, C, _w)
        weight_grf          = cfg(qp_config, "weight_grf",          REQUIRED, C, _w)
        body_mass           = cfg(qp_config, "body_mass",           REQUIRED, C, _w)
        body_inertia        = cfg(qp_config, "body_inertia",        REQUIRED, C, _w)
        desired_body_height = cfg(qp_config, "desired_body_height", REQUIRED, C, _w)
        foot_friction_coef  = cfg(qp_config, "foot_friction_coef",  REQUIRED, C, _w)
        clip_grf            = cfg(qp_config, "clip_grf",            REQUIRED, C, _w)
        use_full_qp         = cfg(qp_config, "use_full_qp",         REQUIRED, C, _w)
        self._stance_kp     = float(cfg(qp_config, "stance_kp",     REQUIRED, C, _w))
        self._stance_kd     = float(cfg(qp_config, "stance_kd",     REQUIRED, C, _w))
        self._acc_lb        = cfg(qp_config, "acc_lb",              REQUIRED, C, _w)
        self._acc_ub        = cfg(qp_config, "acc_ub",              REQUIRED, C, _w)
        self._foot_height_lb = float(cfg(qp_config, "foot_height_lb", REQUIRED, C, _w))
        self._foot_height_ub = float(cfg(qp_config, "foot_height_ub", REQUIRED, C, _w))
        self._grf_z_lb      = float(cfg(qp_config, "grf_z_lb",     REQUIRED, C, _w))
        self._grf_z_ub      = float(cfg(qp_config, "grf_z_ub",     REQUIRED, C, _w))

        flush_warnings(qp_config, C, _w)

        self._clip_grf   = clip_grf
        self._use_full_qp = use_full_qp

        def _vec(v): return to_torch(v, device=self._device)
        def _stack(v): return torch.stack([_vec(v)] * self._num_envs, dim=0)

        self._base_orientation_kp = _stack(base_orientation_kp)
        self._base_orientation_kd = _stack(base_orientation_kd)
        self._base_position_kp    = _stack(base_position_kp)
        self._base_position_kd    = _stack(base_position_kd)

        self._desired_base_orientation_rpy = torch.zeros(
            (self._num_envs, 3), device=self._device
        )
        self._desired_base_position = torch.zeros(
            (self._num_envs, 3), device=self._device
        )
        self._desired_base_position[:, 2] = desired_body_height

        self._desired_linear_velocity     = torch.zeros((self._num_envs, 3), device=self._device)
        self._desired_angular_velocity    = torch.zeros((self._num_envs, 3), device=self._device)
        self._desired_linear_acceleration = torch.zeros((self._num_envs, 3), device=self._device)
        self._desired_angular_acceleration = torch.zeros((self._num_envs, 3), device=self._device)

        self._Wq = to_torch(weight_ddq, device=self._device, dtype=torch.float32)
        self._Wf = to_torch(weight_grf, device=self._device)
        self._foot_friction_coef = foot_friction_coef
        self._inv_mass     = torch.eye(3, device=self._device) / body_mass
        self._inv_inertia  = torch.linalg.inv(
            torch.diag(to_torch(body_inertia, device=self._device))
        )

        logger.info(
            f"[qp] body_mass={body_mass:.3f} kg  "
            f"stance_kp={self._stance_kp}  stance_kd={self._stance_kd}  "
            f"foot_workspace=[{self._foot_height_lb},{self._foot_height_ub}]  "
            f"grf_z=[{self._grf_z_lb},{self._grf_z_ub}]"
        )

    def _solve_joint_torques(self, foot_contact_state, desired_com_ddq):
        self._mass_mat = construct_mass_mat(
            self._robot.foot_positions_in_base_frame,
            foot_contact_state,
            self._inv_mass,
            self._inv_inertia,
            mask_noncontact_legs=not self._use_full_qp,
            device=self._device,
        )
        grf, solved_acc, qp_cost, num_clips = solve_grf(
            self._mass_mat, desired_com_ddq, self._robot.base_rot_mat_t,
            self._Wq, self._Wf, self._foot_friction_coef, self._clip_grf,
            foot_contact_state,
            grf_z_lb=self._grf_z_lb, grf_z_ub=self._grf_z_ub,
            device=self._device,
        )
        all_foot_jacobian = self._robot.all_foot_jacobian
        motor_torques = -torch.bmm(grf[:, None, :], all_foot_jacobian)[:, 0]
        return motor_torques, solved_acc, grf, qp_cost, num_clips

    def compute_joint_command(
        self, foot_contact_state, desired_base_orientation_rpy, desired_base_position,
        desired_foot_position, desired_angular_velocity, desired_linear_velocity,
        desired_foot_velocity, desired_angular_acceleration, desired_linear_acceleration,
        desired_foot_acceleration,
    ):
        desired_acc_body_frame = compute_desired_acc(
            self._robot.base_orientation_rpy, self._robot.base_position,
            self._robot.base_angular_velocity_body_frame,
            self._robot.base_velocity_body_frame,
            desired_base_orientation_rpy, desired_base_position,
            desired_angular_velocity, desired_linear_velocity,
            desired_angular_acceleration, desired_linear_acceleration,
            self._base_position_kp, self._base_position_kd,
            self._base_orientation_kp, self._base_orientation_kd,
            device=self._device,
        )
        desired_acc_body_frame = torch.clip(
            desired_acc_body_frame,
            to_torch(self._acc_lb, device=self._device),
            to_torch(self._acc_ub, device=self._device),
        )

        motor_torques, solved_acc, grf, qp_cost, num_clips = self._solve_joint_torques(
            foot_contact_state, desired_acc_body_frame
        )

        foot_position_local = torch.bmm(
            self._robot.base_rot_mat_t,
            desired_foot_position.transpose(1, 2),
        ).transpose(1, 2)
        foot_position_local[:, :, 2] = torch.clip(
            foot_position_local[:, :, 2],
            min=self._foot_height_lb,
            max=self._foot_height_ub,
        )

        desired_motor_position = self._robot.get_motor_angles_from_foot_positions(
            foot_position_local
        )
        contact_state_expanded = foot_contact_state.repeat_interleave(3, dim=1)

        desired_position = torch.where(
            contact_state_expanded, self._robot.motor_positions, desired_motor_position
        )
        desired_velocity = torch.where(
            contact_state_expanded, self._robot.motor_velocities,
            torch.zeros_like(motor_torques)
        )
        desired_torque = torch.where(
            contact_state_expanded, motor_torques, torch.zeros_like(motor_torques)
        )
        desired_torque = torch.clip(
            desired_torque,
            max=self._robot.motor_group.max_torques,
            min=self._robot.motor_group.min_torques,
        )

        return MotorCommand(
            desired_position=desired_position,
            kp=torch.ones_like(self._robot.motor_group.kps) * self._stance_kp,
            desired_velocity=desired_velocity,
            kd=torch.ones_like(self._robot.motor_group.kds) * self._stance_kd,
            desired_extra_torque=desired_torque,
        ), desired_acc_body_frame, solved_acc, qp_cost, num_clips

    def get_action(self, foot_contact_state, swing_foot_position):
        return self.compute_joint_command(
            foot_contact_state=foot_contact_state,
            desired_base_orientation_rpy=self._desired_base_orientation_rpy,
            desired_base_position=self._desired_base_position,
            desired_foot_position=swing_foot_position,
            desired_angular_velocity=self._desired_angular_velocity,
            desired_linear_velocity=self._desired_linear_velocity,
            desired_foot_velocity=torch.zeros(12),
            desired_angular_acceleration=self._desired_angular_acceleration,
            desired_linear_acceleration=self._desired_linear_acceleration,
            desired_foot_acceleration=torch.zeros(12),
        )

    # ── Properties / setters ──────────────────────────────────────────────────
    @property
    def desired_base_position(self):
        return self._desired_base_position

    @desired_base_position.setter
    def desired_base_position(self, v):
        self._desired_base_position = to_torch(v, device=self._device)

    @property
    def desired_base_orientation_rpy(self):
        return self._desired_base_orientation_rpy

    @desired_base_orientation_rpy.setter
    def desired_base_orientation_rpy(self, v):
        self._desired_base_orientation_rpy = to_torch(v, device=self._device)

    @property
    def desired_linear_velocity(self):
        return self._desired_linear_velocity

    @desired_linear_velocity.setter
    def desired_linear_velocity(self, v):
        self._desired_linear_velocity = to_torch(v, device=self._device)

    @property
    def desired_angular_velocity(self):
        return self._desired_angular_velocity

    @desired_angular_velocity.setter
    def desired_angular_velocity(self, v):
        self._desired_angular_velocity = to_torch(v, device=self._device)

    @property
    def desired_linear_acceleration(self):
        return self._desired_linear_acceleration

    @desired_linear_acceleration.setter
    def desired_linear_acceleration(self, v):
        self._desired_linear_acceleration = to_torch(v, device=self._device)

    @property
    def desired_angular_acceleration(self):
        return self._desired_angular_acceleration

    @desired_angular_acceleration.setter
    def desired_angular_acceleration(self, v):
        self._desired_angular_acceleration = to_torch(v, device=self._device)
