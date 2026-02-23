"""<src/robots/motors.py>Implements motor models for different motor control modes."""
from collections import deque
from dataclasses import dataclass
import enum
from typing import Optional, Sequence, Tuple, Union

from isaacgym.torch_utils import to_torch
from loguru import logger
import torch

_ARRAY = Sequence[float]
_FloatOrArray = Union[float, _ARRAY]


class MotorControlMode(enum.Enum):
    POSITION = 0
    HYBRID   = 1


@dataclass
class MotorCommand:
    desired_position:     torch.Tensor = torch.zeros(12)
    kp:                   torch.Tensor = torch.zeros(12)
    desired_velocity:     torch.Tensor = torch.zeros(12)
    kd:                   torch.Tensor = torch.zeros(12)
    desired_extra_torque: torch.Tensor = torch.zeros(12)


class MotorModel:
    """Describes characteristics of a single motor.
    All arguments required — no defaults. Missing values fail at construction.
    """
    def __init__(
        self,
        name:               str,
        motor_control_mode: MotorControlMode,
        init_position:      float,
        min_position:       float,
        max_position:       float,
        min_velocity:       float,
        max_velocity:       float,
        min_torque:         float,
        max_torque:         float,
        kp:                 float,
        kd:                 float,
    ) -> None:
        self._name               = name
        self._motor_control_mode = motor_control_mode
        self._init_position      = init_position
        self._min_position       = min_position
        self._max_position       = max_position
        self._min_velocity       = min_velocity
        self._max_velocity       = max_velocity
        self._min_torque         = min_torque
        self._max_torque         = max_torque
        self._kp                 = kp
        self._kd                 = kd


class MotorGroup:
    """Models the behaviour of a group of motors.

    SHARED STATE NOTE: torque tensors from convert_to_torque() are consumed
    by robot.step() and read by robot.motor_torques. Do not hold references
    to return values across sim steps — Isaac Gym may invalidate them.

    All args required — no defaults for physics parameters.
    """
    def __init__(
        self,
        device:                           str,
        num_envs:                         int,
        motors:                           Tuple[MotorModel, ...],
        torque_delay_steps:               int,
        enable_realistic_motor_behaviour: bool,
    ):
        self._motors          = motors
        self._num_envs        = num_envs
        self._num_motors      = len(motors)
        self._motor_control_mode = motors[0]._motor_control_mode
        self._device          = device
        self._enable_realistic_motor_behaviour = enable_realistic_motor_behaviour

        if enable_realistic_motor_behaviour:
            logger.info("[motors] Speed-torque curve ENABLED (realistic motor behaviour)")
        else:
            logger.warning(
                "[motors] Speed-torque curve DISABLED "
                "(enable_realistic_motor_behaviour=False)"
            )

        self._strength_ratios = torch.ones(
            (self._num_envs, self._num_motors), device=device
        )
        self._init_positions  = to_torch([m._init_position for m in motors], device=device)
        self._min_positions   = to_torch([m._min_position  for m in motors], device=device)
        self._max_positions   = to_torch([m._max_position  for m in motors], device=device)
        self._min_velocities  = to_torch([m._min_velocity  for m in motors], device=device)
        self._max_velocities  = to_torch([m._max_velocity  for m in motors], device=device)
        self._min_torques     = to_torch([m._min_torque    for m in motors], device=device)
        self._max_torques     = to_torch([m._max_torque    for m in motors], device=device)
        self._kps = to_torch([m._kp for m in motors], device=device)
        self._kds = to_torch([m._kd for m in motors], device=device)

        self._torque_history = deque(maxlen=torque_delay_steps + 1)
        self._torque_history.append(
            torch.zeros((self._num_envs, self._num_motors), device=self._device)
        )
        self._torque_output = torch.zeros(
            (self._num_envs, self._num_motors), device=self._device
        )
        self._true_motor_torque = torch.zeros(
            (self._num_envs, self._num_motors), device=self._device
        )

        logger.debug(
            f"[motors] MotorGroup: {self._num_motors} motors, "
            f"{self._num_envs} envs, delay={torque_delay_steps}, device={device}"
        )
        logger.debug(f"[motors] Torque limits : min={self._min_torques.tolist()}")
        logger.debug(f"[motors]               : max={self._max_torques.tolist()}")
        logger.debug(f"[motors] Vel limits    : min={self._min_velocities.tolist()}")
        logger.debug(f"[motors]               : max={self._max_velocities.tolist()}")

    def _clip_torques(self, desired_torque, current_motor_velocity):
        """Speed-dependent torque clipping (realistic motor behaviour)."""
        torque_ub = torch.where(
            current_motor_velocity < 0,
            self._max_torques,
            self._max_torques * (1 - current_motor_velocity / self._max_velocities),
        )
        torque_lb = torch.where(
            current_motor_velocity < 0,
            self._min_torques * (1 - current_motor_velocity / self._min_velocities),
            self._min_torques,
        )
        return torch.clip(desired_torque, torque_lb, torque_ub)

    def _clip_torques_no_speed_limit(self, desired_torque, current_motor_velocity):
        """Simple torque clipping — max/min only, ignores velocity."""
        return torch.clip(desired_torque, self._min_torques, self._max_torques)

    def convert_to_torque(
        self,
        command:             MotorCommand,
        current_position:    _ARRAY,
        current_velocity:    _ARRAY,
        motor_control_mode:  Optional[MotorControlMode] = None,
    ):
        mode = motor_control_mode or self._motor_control_mode

        if mode == MotorControlMode.POSITION:
            desired_position = command.desired_position
            kp               = self._kps
            desired_velocity = torch.zeros(
                (self._num_envs, self._num_motors), device=self._device
            )
            kd = self._kds
        else:  # HYBRID
            desired_position = command.desired_position
            kp               = command.kp
            desired_velocity = command.desired_velocity
            kd               = command.kd
            self._torque_history.append(command.desired_extra_torque)
            self._torque_output = 0 * self._torque_output + 1. * self._torque_history[0]

        total_torque = (
            kp * (desired_position - current_position)
            + kd * (desired_velocity - current_velocity)
            + self._torque_output
        )

        if self._enable_realistic_motor_behaviour:
            applied_torque = self._clip_torques(total_torque, current_velocity)
        else:
            applied_torque = self._clip_torques_no_speed_limit(total_torque, current_velocity)

        applied_torque *= self._strength_ratios
        return applied_torque, total_torque

    @property
    def motor_control_mode(self):
        return self._motor_control_mode

    @property
    def kps(self):
        return self._kps

    @kps.setter
    def kps(self, value: _FloatOrArray):
        self._kps = torch.ones(self._num_motors, device=self._device) * value

    @property
    def kds(self):
        return self._kds

    @kds.setter
    def kds(self, value: _FloatOrArray):
        self._kds = torch.ones(self._num_motors, device=self._device) * value

    @property
    def strength_ratios(self):
        return self._strength_ratios

    @strength_ratios.setter
    def strength_ratios(self, value: _FloatOrArray):
        self._strength_ratios = (
            torch.ones(self._num_motors, device=self._device)
            * to_torch(value, device=self._device)
        )

    @property
    def init_positions(self):
        return self._init_positions

    @init_positions.setter
    def init_positions(self, value: _ARRAY):
        self._init_positions = value

    @property
    def num_motors(self):
        return self._num_motors

    @property
    def min_positions(self):
        return self._min_positions

    @property
    def max_positions(self):
        return self._max_positions

    @property
    def min_torques(self):
        return self._min_torques

    @property
    def max_torques(self):
        return self._max_torques
