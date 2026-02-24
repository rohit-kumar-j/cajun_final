"""Config for Go2 g2_rotary gait — speed tracking environment.

Structure (being migrated from flat config):
  config.gait      ← gait parameters (complete)
  config.robot     ← robot geometry, motor specs, URDF (NEW — Step 1)
  config.obs       ← observation bounds (TODO — Step 4)
  config.action    ← action bounds (TODO — Step 4)
  config.qp        ← QP controller params (TODO — Step 3)
  config.raibert   ← Raibert controller params (TODO — Step 3)
  config.rewards   ← reward functions and scales (TODO — Step 5)
  config.env       ← termination, episode, env params (TODO — Step 4)

Flat top-level keys below are LEGACY — they remain until their step
migrates them into their section. Do not add new flat keys.
"""
from ml_collections import ConfigDict
import numpy as np
import torch


def get_config():
    config = ConfigDict()

    # =========================================================================
    # config.gait — complete, no changes needed
    # =========================================================================
    gait_config = ConfigDict()
    gait_config.gait_name         = "g2_rotary"
    gait_config.stepping_frequency = 2.0
    # Correct order: [FrontRight, FrontLeft, RearRight, RearLeft]
    gait_config.initial_offset    = np.array(
        [0., 0.15, 0.5, 0.65], dtype=np.float32
    ) * (2 * np.pi)
    gait_config.swing_ratio       = np.array(
        [0.75, 0.75, 0.75, 0.75], dtype=np.float32
    )
    config.gait = gait_config

    # =========================================================================
    # config.robot — NEW (Step 1)
    # All values confirmed against data/go2/urdf/go2.urdf via check_urdf.py
    # =========================================================================
    robot_config = ConfigDict()

    # URDF path — read by UnitreeQuad.__init__, passed to Robot._load_urdf
    robot_config.urdf_path = "data/go1/urdf/go1_u125.urdf"  # switched to Go1 to match baseline

    # Robot standing/spawn height (metres)
    # Used in: jump_env._compute_init_positions, _resample_command,
    #          go1_rewards.height_reward, qp_torque_optimizer (desired_body_height)
    # Was: 0.268 (jump_env), 0.26 (rewards + QP) — unified here
    robot_config.init_height = 0.235

    # Motor hardware latency model
    robot_config.motor_torque_delay_steps           = 5
    robot_config.enable_realistic_motor_behaviour   = True  # False matches baseline: full torque available at all velocities
                                                                  # Set True to test physically accurate Go2 motor model separately

    # Geometry — confirmed from URDF via check_urdf.py
    # Hip joint origins in trunk frame [x, y, z] metres
    robot_config.com_offset = [0.011611, 0.004437, 0.000108]  # Go1 trunk COM offset

    robot_config.hip_positions = [
        [ 0.1881, -0.04675, 0.],  # FR  — Go1
        [ 0.1881,  0.04675, 0.],  # FL  — Go1
        [-0.1881, -0.04675, 0.],  # RR  — Go1
        [-0.1881,  0.04675, 0.],  # RL  — Go1
    ]

    # Hip positions including abduction offset (hip_y + thigh_joint_y)
    # = 0.0465 + 0.0955 = 0.1420 m — confirmed from check_urdf.py positions table
    robot_config.hip_positions_with_abduction = [
        [ 0.1835, -0.131,  0.],   # FR  — Go1 (note: asymmetric y, from go1.py)
        [ 0.1835,  0.122,  0.],   # FL  — Go1
        [-0.1926, -0.131,  0.],   # RR  — Go1
        [-0.1926,  0.122,  0.],   # RL  — Go1
    ]

    # Leg link lengths (metres) — from URDF joint origin z-distances
    robot_config.l_up  = 0.213   # thigh link — same for Go1 and Go2
    robot_config.l_low = 0.233   # calf link — Go1 (Go2 is 0.213)
    robot_config.l_hip = 0.08    # abduction offset — Go1 (Go2 is 0.0955)

    # Foot sphere collision radius — from URDF collision geometry
    # NOTE: go1_rewards.py currently uses 0.02 (wrong) — fix in Step 5
    robot_config.foot_radius = 0.022

    # Link names — must match URDF exactly
    robot_config.feet_names  = ["1_FR_foot",  "2_FL_foot",  "3_RR_foot",  "4_RL_foot"]
    robot_config.calf_names  = ["1_FR_calf",  "2_FL_calf",  "3_RR_calf",  "4_RL_calf"]
    robot_config.thigh_names = ["1_FR_thigh", "2_FL_thigh", "3_RR_thigh", "4_RL_thigh"]

    # Motor specifications — confirmed from URDF via check_urdf.py
    # Order: FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
    #        RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
    #
    # URDF limits (effort = Nm, velocity = rad/s):
    #   hip   : effort=23.7,  velocity=30.1,  position=[-1.0472, +1.0472]
    #   thigh : effort=23.7,  velocity=30.1,  position=[-1.5708,+3.4907] (FR/FL)
    #                                                   [-0.5236,+4.5379] (RR/RL)
    #   calf  : effort=45.43, velocity=15.70, position=[-2.7227,-0.8378]
    #
    # kp/kd: stance gains previously hardcoded as 60/2 in qp_torque_optimizer.
    # Using 40/1 here (motor group defaults); QP stance kp/kd will override
    # via config.qp.stance_kp/kd (to be added in Step 3).
    _HIP_INIT   =  0.0
    _THIGH_INIT =  0.9
    _CALF_INIT  = -1.8
    _KP, _KD    = 100.0, 1.0   # motor model kp (not used in HYBRID mode — stance kp comes from config.qp.stance_kp)

    robot_config.motors = [
        # ── FR ────────────────────────────────────────────────────────────────
        {"name": "1_FR_hip_joint",
         "init_position":  _HIP_INIT,
         "min_position": -0.8029,  "max_position": 0.8029,
         "min_velocity": -30.0,    "max_velocity": 30.0,
         "min_torque":   -23.7,    "max_torque":   23.7,
         "kp": _KP, "kd": _KD},
        {"name": "1_FR_thigh_joint",
         "init_position":  _THIGH_INIT,
         "min_position": -1.0472,  "max_position": 4.1888,
         "min_velocity": -30.0,    "max_velocity": 30.0,
         "min_torque":   -23.7,    "max_torque":   23.7,
         "kp": _KP, "kd": _KD},
        {"name": "1_FR_calf_joint",
         "init_position":  _CALF_INIT,
         "min_position": -2.6965,  "max_position": -0.9163,
         "min_velocity": -20.0,    "max_velocity": 20.0,
         "min_torque":   -35.55,   "max_torque":   35.55,
         "kp": _KP, "kd": _KD},
        # ── FL ────────────────────────────────────────────────────────────────
        {"name": "2_FL_hip_joint",
         "init_position":  _HIP_INIT,
         "min_position": -0.8029,  "max_position": 0.8029,
         "min_velocity": -30.0,    "max_velocity": 30.0,
         "min_torque":   -23.7,    "max_torque":   23.7,
         "kp": _KP, "kd": _KD},
        {"name": "2_FL_thigh_joint",
         "init_position":  _THIGH_INIT,
         "min_position": -1.0472,  "max_position": 4.1888,
         "min_velocity": -30.0,    "max_velocity": 30.0,
         "min_torque":   -23.7,    "max_torque":   23.7,
         "kp": _KP, "kd": _KD},
        {"name": "2_FL_calf_joint",
         "init_position":  _CALF_INIT,
         "min_position": -2.6965,  "max_position": -0.9163,
         "min_velocity": -20.0,    "max_velocity": 20.0,
         "min_torque":   -35.55,   "max_torque":   35.55,
         "kp": _KP, "kd": _KD},
        # ── RR ────────────────────────────────────────────────────────────────
        {"name": "3_RR_hip_joint",
         "init_position":  _HIP_INIT,
         "min_position": -0.8029,  "max_position": 0.8029,
         "min_velocity": -30.0,    "max_velocity": 30.0,
         "min_torque":   -23.7,    "max_torque":   23.7,
         "kp": _KP, "kd": _KD},
        {"name": "3_RR_thigh_joint",       # ← different limits from FR/FL thigh
         "init_position":  _THIGH_INIT,
         "min_position": -1.0472,  "max_position": 4.1888,
         "min_velocity": -30.0,    "max_velocity": 30.0,
         "min_torque":   -23.7,    "max_torque":   23.7,
         "kp": _KP, "kd": _KD},
        {"name": "3_RR_calf_joint",
         "init_position":  _CALF_INIT,
         "min_position": -2.6965,  "max_position": -0.9163,
         "min_velocity": -20.0,    "max_velocity": 20.0,
         "min_torque":   -35.55,   "max_torque":   35.55,
         "kp": _KP, "kd": _KD},
        # ── RL ────────────────────────────────────────────────────────────────
        {"name": "4_RL_hip_joint",
         "init_position":  _HIP_INIT,
         "min_position": -0.8029,  "max_position": 0.8029,
         "min_velocity": -30.0,    "max_velocity": 30.0,
         "min_torque":   -23.7,    "max_torque":   23.7,
         "kp": _KP, "kd": _KD},
        {"name": "4_RL_thigh_joint",       # ← different limits from FR/FL thigh
         "init_position":  _THIGH_INIT,
         "min_position": -1.0472,  "max_position": 4.1888,
         "min_velocity": -30.0,    "max_velocity": 30.0,
         "min_torque":   -23.7,    "max_torque":   23.7,
         "kp": _KP, "kd": _KD},
        {"name": "4_RL_calf_joint",
         "init_position":  _CALF_INIT,
         "min_position": -2.6965,  "max_position": -0.9163,
         "min_velocity": -20.0,    "max_velocity": 20.0,
         "min_torque":   -35.55,   "max_torque":   35.55,
         "kp": _KP, "kd": _KD},
    ]

    config.robot = robot_config


    # =========================================================================
    # config.raibert — NEW (Step 3)
    # =========================================================================
    raibert_config = ConfigDict()
    # desired_base_height: target CoM height used in swing foot trajectory.
    # Mirrors config.env.init_height — single source via config.robot.init_height.
    # NOTE: raibert controller uses this for mid-swing foot height calculation.
    raibert_config.desired_base_height    = 0.41    # matches original RaibertSwingLegController default

    raibert_config.foot_height            = 0.0     # mid-air swing foot height above ground
                                                     # (was 0.15 in original default, g2_rotary overrides to 0.)
    raibert_config.foot_landing_clearance = 0.0     # landing height clearance offset

    # Raibert kp — velocity-dependent gain: kp = kp_base + kp_vel_scale * vel_mag
    # then clamped to [kp_min, kp_max], scaled by alpha from policy
    raibert_config.default_kp    = 0.55
    raibert_config.kp_base       = 0.3
    raibert_config.kp_vel_scale  = 0.05
    raibert_config.kp_min        = 0.3
    raibert_config.kp_max        = 0.65

    config.raibert = raibert_config

    # =========================================================================
    # config.qp — NEW (Step 3)
    # =========================================================================
    qp_config = ConfigDict()

    # PD gains for QP centroidal controller
    qp_config.base_position_kp    = np.array([0.,  0.,  0.])
    qp_config.base_position_kd    = np.array([10., 10., 10.])
    qp_config.base_orientation_kp = np.array([50., 0.,  0.])
    qp_config.base_orientation_kd = np.array([10., 10., 10.])

    # Stance leg PD gains (applied inside compute_joint_command)
    # Were hardcoded as kp=60, kd=2 in the original — now explicit
    qp_config.stance_kp = 30.0
    qp_config.stance_kd = 1.0

    # QP weights
    qp_config.weight_ddq = np.diag([1., 1., 10., 10., 10., 1.])
    qp_config.weight_grf = 1e-4

    # Robot physical properties for QP (inertia is tuned, not raw URDF values)
    # URDF total mass = 15.018 kg — confirmed by check_urdf.py
    # body_inertia is intentionally scaled up from URDF (controller tuning)
    qp_config.body_mass    = 17.17   # Go1 mass (matches go1.urdf; was 15.018 for Go2)
    qp_config.body_inertia = np.array([0.14, 0.35, 0.35]) * 1.5

    # desired_body_height: QP target CoM height — unified with config.robot.init_height
    qp_config.desired_body_height = 0.235

    # Friction and contact
    qp_config.foot_friction_coef = 0.9
    qp_config.clip_grf           = True

    # GRF vertical force bounds (Newtons)
    # 10 N min prevents negative (pulling) contact forces
    # 800 N max limits force spikes
    qp_config.grf_z_lb = 10.0
    qp_config.grf_z_ub = 500.0

    # Body-frame acceleration bounds [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]
    qp_config.acc_lb = [-30., -30., -10., -20., -20., -20.]
    qp_config.acc_ub = [ 30.,  30.,  30.,  20.,  20.,  20.]

    # Foot workspace height limits (metres, in base frame, negative = below base)
    qp_config.foot_height_lb = -0.55
    qp_config.foot_height_ub = -0.10

    qp_config.use_full_qp = False

    config.qp = qp_config


    # =========================================================================
    # config.obs — NEW (Step 4)
    # Observation normalisation bounds for the policy network input
    # =========================================================================
    obs_config = ConfigDict()

    # Robot state observation bounds
    # [base_height, roll*0, pitch, vx, vy*0, vz, ang_vx*0, ang_vy, ang_vyaw,
    #  foot_pos_in_base_frame × 4 legs × 3 axes]
    obs_config.observation_robot_lb = (
        [0., -3.14, -3.14, -4., -4., -10., -3.14, -3.14, -3.14] + [-0.5, -0.5, -0.4] * 4
    )
    obs_config.observation_robot_ub = (
        [0.6, 3.14, 3.14, 8., 4., 10., 3.14, 3.14, 3.14] + [0.5, 0.5, 0.] * 4
    )

    # Task observation bounds [dist_x, dist_y, cos_phase, sin_phase, vel_cmd_xy]
    obs_config.task_lb = [-2., -2., -1., -1., -1.]
    obs_config.task_ub = [ 2.,  2.,  1.,  1.,  1.]

    # Raibert alpha bounds (only used if learn_raibert_alpha=True)
    obs_config.alpha_lb = [0.5]
    obs_config.alpha_ub = [2.0]

    # Height observation bounds (only used if observe_heights=True)
    obs_config.height_obs_lb = -3.
    obs_config.height_obs_ub =  3.

    config.obs = obs_config

    # =========================================================================
    # config.env — NEW (Step 4)
    # Environment, episode, and termination parameters
    # =========================================================================
    env_config = ConfigDict()

    env_config.env_dt             = 0.01
    env_config.episode_length_s   = 20.
    env_config.max_jumps          = 30.
    env_config.velocity_lb        = 0.3   # m/s
    env_config.velocity_ub        = 6.0   # m/s
    env_config.goal_lb            = torch.tensor([0.3, 0.], dtype=torch.float)
    env_config.goal_ub            = torch.tensor([2.0, 0.], dtype=torch.float)

    env_config.motor_strength_ratios    = 1.
    env_config.foot_friction            = 1.
    env_config.use_penetrating_contact  = False
    env_config.use_yaw_feedback         = True
    env_config.yaw_feedback_gain        = 1.0   # was hardcoded as 1 * yaw_err

    env_config.terminate_on_height         = 0.15
    env_config.terminate_on_gravity_z      = 0.5   # was hardcoded in _is_done
    env_config.terminate_on_body_contact   = True
    env_config.terminate_on_limb_contact   = False

    env_config.include_gait_action  = True
    env_config.include_foot_action  = True
    env_config.mirror_foot_action   = False
    env_config.learn_raibert_alpha  = True

    # Action bounds: [step_freq, height, vx, vy, vz, roll, pitch, pitch_rate, yaw_rate]
    #                + foot residuals (4 legs × 3 axes)
    env_config.action_lb = np.array(
        [0.5,  -0.001, -3.,  -0.001, -3.,  -0.001, -0.001, -2.5, -0.001] +
        [-0.1, -0.0001, -0.001] * 4
    )
    env_config.action_ub = np.array(
        [3.999, 0.001,  3.,   0.001,  3.,   0.001,  0.001,  2.5,  0.001] +
        [0.1,   0.0001,  0.2] * 4
    )

    config.env = env_config

    # =========================================================================
    # config.rewards — NEW (Step 5)
    # Reward function scales and tuning parameters
    # =========================================================================
    rewards_config = ConfigDict()

    rewards_config.rewards = [
        ("upright",                      0.02),
        ("contact_consistency",          0.008),
        ("foot_slipping",                0.032),
        ("foot_clearance",               0.008),
        ("out_of_bound_action",          0.01),
        ("knee_contact",                 0.064),
        ("com_distance_to_goal_squared", 0.016),
        ("com_height",                   0.01),
        ("speed_tracking",               0.01),
        ("forward_speed",                0.02),
    ]
    rewards_config.terminal_rewards              = []
    rewards_config.clip_negative_reward          = False
    rewards_config.normalize_reward_by_phase     = True
    rewards_config.clip_negative_terminal_reward = False

    # Reward function thresholds (Step 5 — moved from go1_rewards.py)
    rewards_config.foot_radius             = 0.022  # URDF sphere radius (was 0.02 — wrong)
    rewards_config.foot_clearance_threshold = 0.02
    rewards_config.motor_heat_coefficient  = 0.3
    rewards_config.energy_clip_max         = 2000.
    rewards_config.com_height_reward_max   = 0.5
    rewards_config.limb_force_clip_max      = 10.   # clip for foot_force_reward normalisation
    rewards_config.min_stepping_freq_reward = 2.0   # stepping_freq_reward breakeven (Hz)

    config.rewards = rewards_config

    # =========================================================================
    # LEGACY flat keys — to be migrated into sections in Steps 3-5
    # DO NOT add new flat keys here
    # =========================================================================


    # Action: [step_freq, height, vx, vy, vz, roll, pitch, pitch_rate, yaw_rate]
    #         + foot residuals (×4 legs × 3 axes = 12)





    return config
