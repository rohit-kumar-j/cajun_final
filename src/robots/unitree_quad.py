"""<src/robots/unitree_quad.py>
Vectorised Unitree quadruped robot in Isaac Gym. Supersedes go1.py.

SHARED STATE (gymtorch views into Isaac Gym GPU/CPU buffers):
  _root_states     : actor root state (pos, quat, lin_vel, ang_vel)
  _dof_state       : DOF state (position, velocity per joint)
  _contact_forces  : net contact force per body
  _motor_torques   : DOF force sensor readings
  _jacobian        : Jacobian tensor
  _rigid_body_state: rigid body state (pos, quat, vel per body)
All refreshed each step via _post_physics_step(). Do not cache slices across steps.
"""
from typing import Any, Sequence

from isaacgym.torch_utils import to_torch
from loguru import logger
import ml_collections
import torch

from src.robots.robot import Robot
from src.robots.motors import MotorControlMode, MotorGroup, MotorModel
from src.utilities.config_utils import cfg, flush_warnings, REQUIRED

_ARRAY = Sequence[float]


@torch.jit.script
def motor_angles_from_foot_positions(
    foot_local_positions,
    hip_offset,
    l_up:        float,
    l_low:       float,
    l_hip:       float,
    l_hip_signs,
    device:      str = "cuda",
):
    """Inverse kinematics: foot positions in hip frame → joint angles.

    Args:
        l_up:       thigh link length (metres)         — from config.robot.l_up
        l_low:      calf link length  (metres)         — from config.robot.l_low
        l_hip:      abduction offset  (thigh joint y)  — from config.robot.l_hip
        l_hip_signs: (4,) = [-1, 1, -1, 1] for FR/FL/RR/RL lateral sign
    """
    foot_positions_in_hip_frame = foot_local_positions - hip_offset
    l_hip_vec = l_hip * l_hip_signs

    x = foot_positions_in_hip_frame[:, :, 0]
    y = foot_positions_in_hip_frame[:, :, 1]
    z = foot_positions_in_hip_frame[:, :, 2]

    theta_knee = -torch.arccos(
        torch.clip(
            (x**2 + y**2 + z**2 - l_hip_vec**2 - l_low**2 - l_up**2)
            / (2 * l_low * l_up),
            -1, 1,
        )
    )
    l = torch.sqrt(
        torch.clip(
            l_up**2 + l_low**2 + 2 * l_up * l_low * torch.cos(theta_knee),
            1e-7, 1,
        )
    )
    theta_hip = torch.arcsin(torch.clip(-x / l, -1, 1)) - theta_knee / 2
    c1 = l_hip_vec * y - l * torch.cos(theta_hip + theta_knee / 2) * z
    s1 = l * torch.cos(theta_hip + theta_knee / 2) * y + l_hip_vec * z
    theta_ab = torch.arctan2(s1, c1)

    joint_angles = torch.stack([theta_ab, theta_hip, theta_knee], dim=2)
    return joint_angles.reshape((-1, 12))


class UnitreeQuad(Robot):
    """Unitree quadruped in Isaac Gym simulation. All geometry from config.robot.

    Args:
        sim:           Isaac Gym sim handle
        viewer:        Isaac Gym viewer (None if headless)
        sim_config:    result of sim_config.get_config()
        num_envs:      number of parallel environments
        init_positions:(num_envs, 3) spawn positions
        motor_control_mode: MotorControlMode.HYBRID or POSITION
        robot_config:  config.robot sub-config
    """
    def __init__(
        self,
        sim:                Any,
        viewer:             Any,
        sim_config:         ml_collections.ConfigDict,
        num_envs:           int,
        init_positions:     _ARRAY,
        motor_control_mode: MotorControlMode,
        robot_config:       ml_collections.ConfigDict,
    ):
        device = sim_config.sim_device
        _w: dict = {}  # warning flags → written to config.robot.warnings

        # ── Read all robot config values ──────────────────────────────────────
        urdf_path        = cfg(robot_config, "urdf_path",        REQUIRED,        "config.robot", _w)
        init_height      = cfg(robot_config, "init_height",      REQUIRED,           "config.robot", _w)
        torque_delay     = cfg(robot_config, "motor_torque_delay_steps", REQUIRED,   "config.robot", _w)
        enable_realistic = cfg(robot_config, "enable_realistic_motor_behaviour", REQUIRED, "config.robot", _w)
        com_offset_raw   = cfg(robot_config, "com_offset",   REQUIRED,   "config.robot", _w)
        hip_pos_raw      = cfg(robot_config, "hip_positions", REQUIRED,
                                # [[ 0.1934,-0.0465,0.],[ 0.1934, 0.0465,0.],
                                #  [-0.1934,-0.0465,0.],[-0.1934, 0.0465,0.]],
                                "config.robot", _w)
        hip_pos_abd_raw  = cfg(robot_config, "hip_positions_with_abduction", REQUIRED,
                                # [[ 0.1934,-0.1420,0.],[ 0.1934, 0.1420,0.],
                                #  [-0.1934,-0.1420,0.],[-0.1934, 0.1420,0.]],
                                "config.robot", _w)
        self._l_up       = float(cfg(robot_config, "l_up",  REQUIRED,  "config.robot", _w))
        self._l_low      = float(cfg(robot_config, "l_low", REQUIRED,  "config.robot", _w))
        self._l_hip      = float(cfg(robot_config, "l_hip", REQUIRED, "config.robot", _w))
        feet_names       = list(cfg(robot_config, "feet_names",
                                     ["1_FR_foot","2_FL_foot","3_RR_foot","4_RL_foot"],
                                     "config.robot", _w))
        calf_names       = list(cfg(robot_config, "calf_names",
                                     ["1_FR_calf","2_FL_calf","3_RR_calf","4_RL_calf"],
                                     "config.robot", _w))
        thigh_names      = list(cfg(robot_config, "thigh_names",
                                     ["1_FR_thigh","2_FL_thigh","3_RR_thigh","4_RL_thigh"],
                                     "config.robot", _w))
        motor_specs      = cfg(robot_config, "motors", REQUIRED, "config.robot", _w)

        flush_warnings(robot_config, "config.robot", _w)

        # ── Startup summary ───────────────────────────────────────────────────
        missing = [k for k, v in _w.items() if not v]
        logger.info("=" * 52)
        logger.info("[unitree_quad] ROBOT CONFIG SUMMARY")
        logger.info("=" * 52)
        logger.info(f"  urdf_path        : {urdf_path}")
        logger.info(f"  init_height      : {init_height} m")
        logger.info(f"  torque_delay     : {torque_delay} steps")
        logger.info(f"  realistic motors : {enable_realistic}")
        logger.info(f"  l_up / l_low     : {self._l_up} / {self._l_low} m")
        logger.info(f"  l_hip            : {self._l_hip} m")
        if missing:
            logger.warning(f"  AUTO-FILLED (missing from config): {missing}")
        else:
            logger.success("  All config.robot values explicitly set")
        logger.info("=" * 52)

        # ── Build geometry ────────────────────────────────────────────────────
        # com_offset convention (matches original go1.py):
        #   hip_offset = hip_joint_positions - com_offset
        # i.e. hip positions expressed in the frame centred at the CoM.
        # config.robot.com_offset stores the raw URDF inertial origin (positive x forward).
        # We negate it here: subtracting CoM offset shifts hip positions relative to CoM.
        com_offset       = -to_torch(com_offset_raw, device=device)
        self._hip_offset = to_torch(hip_pos_raw, device=device) + com_offset
        hip_single  = to_torch(hip_pos_abd_raw, device=device)
        self._hip_positions_in_body_frame = torch.stack([hip_single] * num_envs, dim=0)
        self._l_hip_signs = torch.tensor([-1., 1., -1., 1.], device=device)

        # ── Build MotorGroup ──────────────────────────────────────────────────
        motor_models = []
        for spec in motor_specs:
            try:
                motor_models.append(MotorModel(
                    name               = spec["name"],
                    motor_control_mode = motor_control_mode,
                    init_position      = spec["init_position"],
                    min_position       = spec["min_position"],
                    max_position       = spec["max_position"],
                    min_velocity       = spec["min_velocity"],
                    max_velocity       = spec["max_velocity"],
                    min_torque         = spec["min_torque"],
                    max_torque         = spec["max_torque"],
                    kp                 = spec["kp"],
                    kd                 = spec["kd"],
                ))
            except KeyError as exc:
                logger.error(
                    f"[unitree_quad] Motor spec missing key {exc} "
                    f"for joint '{spec.get('name','?')}'"
                )
                raise

        motors = MotorGroup(
            device                           = device,
            num_envs                         = num_envs,
            motors                           = tuple(motor_models),
            torque_delay_steps               = torque_delay,
            enable_realistic_motor_behaviour = enable_realistic,
        )

        super().__init__(
            sim            = sim,
            viewer         = viewer,
            num_envs       = num_envs,
            init_positions = init_positions,
            urdf_path      = urdf_path,
            sim_config     = sim_config,
            motors         = motors,
            feet_names     = feet_names,
            calf_names     = calf_names,
            thigh_names    = thigh_names,
        )
        logger.info("[unitree_quad] Robot fully initialised")

    @property
    def hip_positions_in_body_frame(self):
        return self._hip_positions_in_body_frame

    @property
    def hip_offset(self):
        return self._hip_offset

    def get_motor_angles_from_foot_positions(self, foot_local_positions):
        return motor_angles_from_foot_positions(
            foot_local_positions,
            self.hip_offset,
            l_up        = self._l_up,
            l_low       = self._l_low,
            l_hip       = self._l_hip,
            l_hip_signs = self._l_hip_signs,
            device      = self._device,
        )
