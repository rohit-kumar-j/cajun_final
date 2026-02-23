"""
Phase Gait Visualizer

Visualizes the gait pattern. Based on working visualizer.
Standalone file with embedded PhaseGaitGenerator class.
Search for "# CONFIG:" comments to find configurable values.
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir
while project_root != '/':
    if os.path.isdir(os.path.join(project_root, 'src')):
        break
    project_root = os.path.dirname(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from absl import app
from absl import flags
from isaacgym import gymapi, gymutil, gymtorch
import numpy as np
from tqdm import tqdm
import torch

from src.configs.defaults import sim_config
from src.controllers import raibert_swing_leg_controller
from src.robots import go1
from src.robots.motors import MotorControlMode

flags.DEFINE_integer("num_envs", 1, "Number of environments.")
flags.DEFINE_float("total_time_secs", 120., "Total time.")
flags.DEFINE_bool("use_gpu", True, "Use GPU.")
flags.DEFINE_bool("show_gui", True, "Show GUI.")
FLAGS = flags.FLAGS

# ============================================================================
# CONFIG VALUES - Search "# CONFIG:" to find all configurable parameters
# ============================================================================

# CONFIG: Gait parameters
GAIT_INITIAL_OFFSET = np.array([0.0, np.pi, np.pi, 0.0])  # Trot gait
GAIT_SWING_RATIO = np.array([0.6, 0.6, 0.6, 0.6])  # 60% swing, 40% stance
GAIT_STEPPING_FREQUENCY = 1.5  # Hz

# CONFIG: Swing foot parameters
SWING_FOOT_HEIGHT = 0.1 + 0.15  # Max height during swing (m)
SWING_FOOT_LANDING_CLEARANCE = 0.0

# CONFIG: Robot body parameters
FIXED_HEIGHT = 0.26  # Standard Go1 standing height

# CONFIG: Velocity parameters (initial values)
INITIAL_FAKE_CURRENT_VX = 0.0  # Initial "fake" current velocity (m/s)
INITIAL_DESIRED_VX = 0.0       # Initial desired velocity (m/s)
VELOCITY_INCREMENT = 0.25       # How much to change velocity per keypress (m/s)
MAX_VELOCITY = 5.0              # Maximum velocity (m/s)
MIN_VELOCITY = -2.0             # Minimum velocity (m/s)

# CONFIG: Fake velocity for visualization (set higher than DESIRED_VX to see high-speed gait)
# This fools the Raibert controller into computing foot placements for high speeds
# while the robot stays in place for easy visualization
FAKE_VELOCITY = 3.0  # m/s - change this to visualize different speeds
USE_FAKE_VELOCITY = True  # Set to False to use real velocity ramp

# CONFIG: Visualization colors (RGB)
FOOT_COLORS_STANCE = [
    (0.8, 0.2, 0.2),  # FL: Dark Red (stance)
    (0.2, 0.8, 0.2),  # FR: Dark Green (stance)
    (0.2, 0.2, 0.8),  # RL: Dark Blue (stance)
    (0.8, 0.8, 0.2),  # RR: Dark Yellow (stance)
]
FOOT_COLORS_SWING = [
    (1.0, 0.5, 0.5),  # FL: Light Red (swing)
    (0.5, 1.0, 0.5),  # FR: Light Green (swing)
    (0.5, 0.5, 1.0),  # RL: Light Blue (swing)
    (1.0, 1.0, 0.5),  # RR: Light Yellow (swing)
]


def draw_foot_spheres(gym, viewer, foot_positions, contact_state, env_idx=0):
    """Draw wireframe spheres at foot positions."""
    gym.clear_lines(viewer)
    
    for leg_idx in range(4):
        pos = foot_positions[env_idx, leg_idx].cpu().numpy()
        in_stance = contact_state[env_idx, leg_idx].item()
        
        if in_stance:
            color = FOOT_COLORS_STANCE[leg_idx]
        else:
            color = FOOT_COLORS_SWING[leg_idx]
        
        # Draw wireframe sphere
        radius = 0.03
        segments = 12
        
        for plane in range(3):
            points = []
            for i in range(segments + 1):
                angle = 2 * np.pi * i / segments
                if plane == 0:  # XY plane
                    p = [pos[0] + radius * np.cos(angle), 
                         pos[1] + radius * np.sin(angle), 
                         pos[2]]
                elif plane == 1:  # XZ plane
                    p = [pos[0] + radius * np.cos(angle), 
                         pos[1], 
                         pos[2] + radius * np.sin(angle)]
                else:  # YZ plane
                    p = [pos[0], 
                         pos[1] + radius * np.cos(angle), 
                         pos[2] + radius * np.sin(angle)]
                points.append(p)
            
            for i in range(len(points) - 1):
                gym.add_lines(
                    viewer, None, 1,
                    [points[i][0], points[i][1], points[i][2],
                     points[i+1][0], points[i+1][1], points[i+1][2]],
                    [color[0], color[1], color[2]]
                )


# ============================================================================
# EMBEDDED PHASE GAIT GENERATOR
# ============================================================================
class PhaseGaitGenerator:
    """Computes desired gait based on leg phases."""
    
    def __init__(self, robot, initial_offset, swing_ratio, stepping_frequency):
        self._robot = robot
        self._num_envs = robot.num_envs
        self._device = robot._device
        
        from isaacgym.torch_utils import to_torch
        self._initial_offset = to_torch(initial_offset, device=self._device)
        self._swing_ratio = to_torch(swing_ratio, device=self._device)
        self._stepping_frequency_base = stepping_frequency
        
        self.reset()
    
    def reset(self):
        self._current_phase = torch.stack(
            [self._initial_offset] * self._num_envs, axis=0
        ).to(self._device)
        
        self._stepping_frequency = torch.ones(
            self._num_envs, device=self._device
        ) * self._stepping_frequency_base
        
        self._swing_cutoff = torch.ones(
            (self._num_envs, 4), device=self._device
        ) * 2 * torch.pi * (1 - self._swing_ratio)
        
        self._prev_frame_robot_time = self._robot.time_since_reset.clone()
        self._first_stance_seen = torch.zeros(
            (self._num_envs, 4), dtype=torch.bool, device=self._device
        )
    
    def reset_idx(self, env_ids):
        self._current_phase[env_ids] = self._initial_offset
        self._stepping_frequency[env_ids] = self._stepping_frequency_base
        self._swing_cutoff[env_ids] = 2 * torch.pi * (1 - self._swing_ratio)
        self._prev_frame_robot_time[env_ids] = self._robot.time_since_reset[env_ids]
        self._first_stance_seen[env_ids] = 0
    
    def update(self):
        current_robot_time = self._robot.time_since_reset
        delta_t = current_robot_time - self._prev_frame_robot_time
        self._prev_frame_robot_time = current_robot_time.clone()
        self._current_phase += 2 * torch.pi * self._stepping_frequency[:, None] * delta_t[:, None]
    
    @property
    def desired_contact_state(self):
        modulated_phase = torch.remainder(self._current_phase + 2 * torch.pi, 2 * torch.pi)
        raw_contact = torch.where(modulated_phase > self._swing_cutoff, False, True)
        self._first_stance_seen = torch.logical_or(self._first_stance_seen, raw_contact)
        return torch.where(self._first_stance_seen, raw_contact, torch.ones_like(raw_contact))
    
    @property
    def normalized_phase(self):
        modulated_phase = torch.remainder(self._current_phase + 2 * torch.pi, 2 * torch.pi)
        return torch.where(
            modulated_phase < self._swing_cutoff,
            modulated_phase / self._swing_cutoff,
            (modulated_phase - self._swing_cutoff) / (2 * torch.pi - self._swing_cutoff)
        )
    
    @property
    def stance_duration(self):
        return self._swing_cutoff / (2 * torch.pi * self._stepping_frequency[:, None])
    
    @property
    def stepping_frequency(self):
        return self._stepping_frequency
    
    @stepping_frequency.setter
    def stepping_frequency(self, new_frequency):
        self._stepping_frequency = new_frequency


# ============================================================================
# SIMULATION SETUP
# ============================================================================
def create_sim(sim_conf):
    gym = gymapi.acquire_gym()
    _, sim_device_id = gymutil.parse_device_str(sim_conf.sim_device)
    graphics_device_id = sim_device_id if sim_conf.show_gui else -1
    sim = gym.create_sim(sim_device_id, graphics_device_id,
                         sim_conf.physics_engine, sim_conf.sim_params)
    viewer = None
    if sim_conf.show_gui:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "QUIT")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "PAUSE")
        # Velocity controls
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "CURRENT_VEL_UP")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "CURRENT_VEL_DOWN")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "DESIRED_VEL_UP")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "DESIRED_VEL_DOWN")
        # Frequency controls
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP, "FREQ_UP")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN, "FREQ_DOWN")
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 1.0
    gym.add_ground(sim, plane_params)
    return gym, sim, viewer


def get_init_positions(num_envs, device):
    init_positions = torch.zeros((num_envs, 3), device=device)
    init_positions[:, 2] = 0.34
    return init_positions


def main(argv):
    del argv
    
    sim_conf = sim_config.get_config(use_gpu=FLAGS.use_gpu, show_gui=FLAGS.show_gui)
    gym, sim, viewer = create_sim(sim_conf)
    
    robot = go1.Go1(
        num_envs=FLAGS.num_envs,
        init_positions=get_init_positions(FLAGS.num_envs, sim_conf.sim_device),
        sim=sim,
        viewer=viewer,
        sim_config=sim_conf,
        motor_control_mode=MotorControlMode.POSITION
    )
    
    # Use embedded PhaseGaitGenerator
    gait_generator = PhaseGaitGenerator(
        robot=robot,
        initial_offset=GAIT_INITIAL_OFFSET,       # CONFIG: Phase offsets
        swing_ratio=GAIT_SWING_RATIO,             # CONFIG: Swing ratio
        stepping_frequency=GAIT_STEPPING_FREQUENCY # CONFIG: Stepping frequency
    )
    
    # Use the actual RaibertSwingLegController from your project
    swing_leg_controller = raibert_swing_leg_controller.RaibertSwingLegController(
        robot, gait_generator,
        foot_landing_clearance=SWING_FOOT_LANDING_CLEARANCE,  # CONFIG
        foot_height=SWING_FOOT_HEIGHT                          # CONFIG
    )
    
    robot.reset()
    
    dt = sim_conf.dt * sim_conf.action_repeat
    desired_velocity = torch.zeros((FLAGS.num_envs, 3), device=robot.device)
    env_ids = torch.arange(FLAGS.num_envs, device=robot.device, dtype=torch.int32)
    
    # Track stance foot positions
    stance_foot_world_positions = torch.zeros((FLAGS.num_envs, 4, 3), device=robot.device)
    prev_contact_state = torch.zeros((FLAGS.num_envs, 4), dtype=torch.bool, device=robot.device)
    
    # Fake velocity controls (what we tell the Raibert controller)
    fake_current_vx = INITIAL_FAKE_CURRENT_VX
    fake_desired_vx = INITIAL_DESIRED_VX
    current_frequency = GAIT_STEPPING_FREQUENCY
    paused = False
    
    print("=" * 70)
    print("Phase Gait Visualizer - VELOCITY FAKER MODE")
    print("=" * 70)
    print(f"Gait frequency: {GAIT_STEPPING_FREQUENCY} Hz")
    print(f"Swing ratio: {GAIT_SWING_RATIO}")
    print(f"Phase offsets: {GAIT_INITIAL_OFFSET}")
    print("=" * 70)
    print("KEYBOARD CONTROLS:")
    print("  W/S  - Increase/Decrease FAKE current velocity")
    print("  D/A  - Increase/Decrease desired velocity")
    print("  UP/DOWN - Increase/Decrease stepping frequency")
    print("  SPACE - Pause/Resume")
    print("  ESC - Quit")
    print("=" * 70)
    print(f"Initial: Current V = {fake_current_vx:.2f} m/s, Desired V = {fake_desired_vx:.2f} m/s")
    print("=" * 70)
    
    pbar = tqdm(total=FLAGS.total_time_secs)
    steps = 0
    
    with torch.inference_mode():
        while robot.time_since_reset[0] <= FLAGS.total_time_secs:
            # Poll keyboard events
            if FLAGS.show_gui:
                for evt in gym.query_viewer_action_events(viewer):
                    if evt.action == "QUIT" and evt.value > 0:
                        print("\nQuit requested")
                        pbar.close()
                        return
                    if evt.action == "PAUSE" and evt.value > 0:
                        paused = not paused
                        print(f"\n{'PAUSED' if paused else 'RESUMED'}")
                    if evt.action == "CURRENT_VEL_UP" and evt.value > 0:
                        fake_current_vx = min(fake_current_vx + VELOCITY_INCREMENT, MAX_VELOCITY)
                        print(f"\nCurrent V: {fake_current_vx:.2f} m/s | Desired V: {fake_desired_vx:.2f} m/s | Freq: {current_frequency:.2f} Hz")
                    if evt.action == "CURRENT_VEL_DOWN" and evt.value > 0:
                        fake_current_vx = max(fake_current_vx - VELOCITY_INCREMENT, MIN_VELOCITY)
                        print(f"\nCurrent V: {fake_current_vx:.2f} m/s | Desired V: {fake_desired_vx:.2f} m/s | Freq: {current_frequency:.2f} Hz")
                    if evt.action == "DESIRED_VEL_UP" and evt.value > 0:
                        fake_desired_vx = min(fake_desired_vx + VELOCITY_INCREMENT, MAX_VELOCITY)
                        print(f"\nCurrent V: {fake_current_vx:.2f} m/s | Desired V: {fake_desired_vx:.2f} m/s | Freq: {current_frequency:.2f} Hz")
                    if evt.action == "DESIRED_VEL_DOWN" and evt.value > 0:
                        fake_desired_vx = max(fake_desired_vx - VELOCITY_INCREMENT, MIN_VELOCITY)
                        print(f"\nCurrent V: {fake_current_vx:.2f} m/s | Desired V: {fake_desired_vx:.2f} m/s | Freq: {current_frequency:.2f} Hz")
                    if evt.action == "FREQ_UP" and evt.value > 0:
                        current_frequency = min(current_frequency + 0.1, 4.0)
                        gait_generator._stepping_frequency[:] = current_frequency
                        gait_generator._swing_cutoff[:] = 2 * torch.pi * (1 - gait_generator._swing_ratio)
                        print(f"\nCurrent V: {fake_current_vx:.2f} m/s | Desired V: {fake_desired_vx:.2f} m/s | Freq: {current_frequency:.2f} Hz")
                    if evt.action == "FREQ_DOWN" and evt.value > 0:
                        current_frequency = max(current_frequency - 0.1, 0.5)
                        gait_generator._stepping_frequency[:] = current_frequency
                        gait_generator._swing_cutoff[:] = 2 * torch.pi * (1 - gait_generator._swing_ratio)
                        print(f"\nCurrent V: {fake_current_vx:.2f} m/s | Desired V: {fake_desired_vx:.2f} m/s | Freq: {current_frequency:.2f} Hz")
            
            if paused:
                if FLAGS.show_gui:
                    robot.render()
                continue
            
            t = robot.time_since_reset[0].item()
            
            # Set the desired velocity for the swing leg controller
            desired_velocity[:, 0] = fake_desired_vx
            
            # ===== KINEMATIC MODE: Direct joint position control =====
            
            # Fix base position (robot stays stationary, but we fake velocity)
            robot._root_states[:, 0] = 0.0
            robot._root_states[:, 1] = 0.0
            robot._root_states[:, 2] = FIXED_HEIGHT
            robot._root_states[:, 3:7] = torch.tensor([0., 0., 0., 1.], device=robot.device)
            robot._root_states[:, 7] = 0.0  # Actual velocity is 0 (stationary)
            robot._root_states[:, 8:13] = 0.0
            gym.set_actor_root_state_tensor_indexed(
                sim, gymtorch.unwrap_tensor(robot._root_states),
                gymtorch.unwrap_tensor(env_ids), len(env_ids))
            gym.refresh_actor_root_state_tensor(sim)
            robot._post_physics_step()
            
            gait_generator.update()
            
            # ===== FAKE THE VELOCITY FOR RAIBERT CONTROLLER =====
            # Save the real velocity
            real_base_velocity_world = robot.base_vel.clone()
            
            # Set fake velocity (this is what Raibert controller will see)
            robot.base_vel[:, 0] = fake_current_vx
            robot.base_vel[:, 1] = 0.0
            robot.base_vel[:, 2] = 0.0
            
            # Also need to fake base_velocity_world_frame if it's a separate property
            # The Raibert controller uses self._robot.base_velocity_world_frame
            real_base_velocity_world_frame = robot.base_velocity_world_frame.clone()
            
            # Update swing leg controller with faked velocity
            swing_leg_controller.update(desired_velocity)
            
            # Restore real velocity
            robot.base_vel[:] = real_base_velocity_world
            
            # ===== END VELOCITY FAKING =====
            
            contact_state = gait_generator.desired_contact_state
            
            # Detect touchdown
            just_touched_down = contact_state & ~prev_contact_state
            current_foot_world = robot.foot_positions_in_world_frame
            for leg_idx in range(4):
                mask = just_touched_down[:, leg_idx]
                if mask.any():
                    stance_foot_world_positions[mask, leg_idx] = current_foot_world[mask, leg_idx]
            
            # ===== SIMULATE STANCE FOOT DRIFT =====
            # When faking velocity, stance feet should drift backward relative to body
            # as if the body were actually moving forward
            # drift = -velocity * dt (negative because feet move backward relative to body)
            stance_drift = torch.zeros((FLAGS.num_envs, 4, 3), device=robot.device)
            stance_drift[:, :, 0] = -fake_current_vx * dt  # X drift (backward)
            
            # Apply drift to stance feet (only feet currently in stance)
            for leg_idx in range(4):
                in_stance = contact_state[:, leg_idx]
                if in_stance.any():
                    stance_foot_world_positions[in_stance, leg_idx, 0] += stance_drift[in_stance, leg_idx, 0]
            
            # Swing feet from Raibert controller
            swing_foot_positions_world = swing_leg_controller.desired_foot_positions
            
            # Convert to base frame
            base_pos = robot._root_states[:, :3]
            
            stance_foot_relative = stance_foot_world_positions - base_pos[:, None, :]
            stance_foot_base = torch.bmm(
                robot.base_rot_mat_t,
                stance_foot_relative.transpose(1, 2)
            ).transpose(1, 2)
            
            swing_foot_base = torch.bmm(
                robot.base_rot_mat_t,
                swing_foot_positions_world.transpose(1, 2)
            ).transpose(1, 2)
            
            target_foot_base = torch.where(
                contact_state[:, :, None].expand(-1, -1, 3),
                stance_foot_base,
                swing_foot_base
            )
            
            target_foot_base[:, :, 2] = torch.clip(target_foot_base[:, :, 2], min=-0.35, max=-0.05)
            
            joint_positions = robot.get_motor_angles_from_foot_positions(target_foot_base)
            
            # Directly set joint positions
            robot._dof_state.view(FLAGS.num_envs, 12, 2)[..., 0] = joint_positions
            robot._dof_state.view(FLAGS.num_envs, 12, 2)[..., 1] = 0.0
            gym.set_dof_state_tensor_indexed(
                sim, gymtorch.unwrap_tensor(robot._dof_state),
                gymtorch.unwrap_tensor(env_ids), len(env_ids))
            
            gym.simulate(sim)
            gym.refresh_dof_state_tensor(sim)
            gym.refresh_actor_root_state_tensor(sim)
            gym.refresh_rigid_body_state_tensor(sim)
            robot._post_physics_step()
            
            prev_contact_state = contact_state.clone()
            robot._time_since_reset += dt
            
            steps += 1
            pbar.update(dt)
            
            if steps % 100 == 0:
                contact = contact_state[0].cpu().numpy().astype(int)
                phase = gait_generator.normalized_phase[0].cpu().numpy()
                delta_v = fake_current_vx - fake_desired_vx
                print(f"t={t:.2f}s | V_curr={fake_current_vx:.2f} | V_des={fake_desired_vx:.2f} | ΔV={delta_v:.2f} | freq={current_frequency:.2f} | contact={contact}")
            
            if FLAGS.show_gui:
                # Draw foot position spheres
                current_foot_world = robot.foot_positions_in_world_frame
                draw_foot_spheres(gym, viewer, current_foot_world, contact_state)
                robot.render()
    
    pbar.close()
    print("Done.")


if __name__ == "__main__":
    app.run(main)
#
# """
# Phase Gait Visualizer
#
# Visualizes the gait pattern. Based on working visualizer.
# Standalone file with embedded PhaseGaitGenerator class.
# Search for "# CONFIG:" comments to find configurable values.
# """
# import os
# import sys
#
# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = script_dir
# while project_root != '/':
#     if os.path.isdir(os.path.join(project_root, 'src')):
#         break
#     project_root = os.path.dirname(project_root)
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
#
# from absl import app
# from absl import flags
# from isaacgym import gymapi, gymutil, gymtorch
# import numpy as np
# from tqdm import tqdm
# import torch
#
# from src.configs.defaults import sim_config
# from src.controllers import raibert_swing_leg_controller
# from src.robots import go1
# from src.robots.motors import MotorControlMode
#
# flags.DEFINE_integer("num_envs", 1, "Number of environments.")
# flags.DEFINE_float("total_time_secs", 120., "Total time.")
# flags.DEFINE_bool("use_gpu", True, "Use GPU.")
# flags.DEFINE_bool("show_gui", True, "Show GUI.")
# FLAGS = flags.FLAGS
#
# # ============================================================================
# # CONFIG VALUES - Search "# CONFIG:" to find all configurable parameters
# # ============================================================================
#
# # CONFIG: Gait parameters
# GAIT_INITIAL_OFFSET = np.array([0.0, np.pi, np.pi, 0.0])  # Trot gait
# GAIT_SWING_RATIO = np.array([0.6, 0.6, 0.6, 0.6])  # 60% swing, 40% stance
# GAIT_STEPPING_FREQUENCY = 1.5  # Hz
#
# # CONFIG: Swing foot parameters
# SWING_FOOT_HEIGHT = 0.1 + 0.15  # Max height during swing (m)
# SWING_FOOT_LANDING_CLEARANCE = 0.0
#
# # CONFIG: Robot body parameters
# FIXED_HEIGHT = 0.39  # Standard Go1 standing height
#
# # CONFIG: Velocity parameters (initial values)
# INITIAL_FAKE_CURRENT_VX = 0.0  # Initial "fake" current velocity (m/s)
# INITIAL_DESIRED_VX = 0.0       # Initial desired velocity (m/s)
# VELOCITY_INCREMENT = 0.25       # How much to change velocity per keypress (m/s)
# MAX_VELOCITY = 5.0              # Maximum velocity (m/s)
# MIN_VELOCITY = -2.0             # Minimum velocity (m/s)
#
# # CONFIG: Fake velocity for visualization (set higher than DESIRED_VX to see high-speed gait)
# # This fools the Raibert controller into computing foot placements for high speeds
# # while the robot stays in place for easy visualization
# FAKE_VELOCITY = 3.0  # m/s - change this to visualize different speeds
# USE_FAKE_VELOCITY = True  # Set to False to use real velocity ramp
#
# # CONFIG: Visualization colors (RGB)
# FOOT_COLORS_STANCE = [
#     (0.8, 0.2, 0.2),  # FL: Dark Red (stance)
#     (0.2, 0.8, 0.2),  # FR: Dark Green (stance)
#     (0.2, 0.2, 0.8),  # RL: Dark Blue (stance)
#     (0.8, 0.8, 0.2),  # RR: Dark Yellow (stance)
# ]
# FOOT_COLORS_SWING = [
#     (1.0, 0.5, 0.5),  # FL: Light Red (swing)
#     (0.5, 1.0, 0.5),  # FR: Light Green (swing)
#     (0.5, 0.5, 1.0),  # RL: Light Blue (swing)
#     (1.0, 1.0, 0.5),  # RR: Light Yellow (swing)
# ]
#
#
# def draw_foot_spheres(gym, viewer, foot_positions, contact_state, env_idx=0):
#     """Draw wireframe spheres at foot positions."""
#     gym.clear_lines(viewer)
#     
#     for leg_idx in range(4):
#         pos = foot_positions[env_idx, leg_idx].cpu().numpy()
#         in_stance = contact_state[env_idx, leg_idx].item()
#         
#         if in_stance:
#             color = FOOT_COLORS_STANCE[leg_idx]
#         else:
#             color = FOOT_COLORS_SWING[leg_idx]
#         
#         # Draw wireframe sphere
#         radius = 0.03
#         segments = 12
#         
#         for plane in range(3):
#             points = []
#             for i in range(segments + 1):
#                 angle = 2 * np.pi * i / segments
#                 if plane == 0:  # XY plane
#                     p = [pos[0] + radius * np.cos(angle), 
#                          pos[1] + radius * np.sin(angle), 
#                          pos[2]]
#                 elif plane == 1:  # XZ plane
#                     p = [pos[0] + radius * np.cos(angle), 
#                          pos[1], 
#                          pos[2] + radius * np.sin(angle)]
#                 else:  # YZ plane
#                     p = [pos[0], 
#                          pos[1] + radius * np.cos(angle), 
#                          pos[2] + radius * np.sin(angle)]
#                 points.append(p)
#             
#             for i in range(len(points) - 1):
#                 gym.add_lines(
#                     viewer, None, 1,
#                     [points[i][0], points[i][1], points[i][2],
#                      points[i+1][0], points[i+1][1], points[i+1][2]],
#                     [color[0], color[1], color[2]]
#                 )
#
#
# # ============================================================================
# # EMBEDDED PHASE GAIT GENERATOR
# # ============================================================================
# class PhaseGaitGenerator:
#     """Computes desired gait based on leg phases."""
#     
#     def __init__(self, robot, initial_offset, swing_ratio, stepping_frequency):
#         self._robot = robot
#         self._num_envs = robot.num_envs
#         self._device = robot._device
#         
#         from isaacgym.torch_utils import to_torch
#         self._initial_offset = to_torch(initial_offset, device=self._device)
#         self._swing_ratio = to_torch(swing_ratio, device=self._device)
#         self._stepping_frequency_base = stepping_frequency
#         
#         self.reset()
#     
#     def reset(self):
#         self._current_phase = torch.stack(
#             [self._initial_offset] * self._num_envs, axis=0
#         ).to(self._device)
#         
#         self._stepping_frequency = torch.ones(
#             self._num_envs, device=self._device
#         ) * self._stepping_frequency_base
#         
#         self._swing_cutoff = torch.ones(
#             (self._num_envs, 4), device=self._device
#         ) * 2 * torch.pi * (1 - self._swing_ratio)
#         
#         self._prev_frame_robot_time = self._robot.time_since_reset.clone()
#         self._first_stance_seen = torch.zeros(
#             (self._num_envs, 4), dtype=torch.bool, device=self._device
#         )
#     
#     def reset_idx(self, env_ids):
#         self._current_phase[env_ids] = self._initial_offset
#         self._stepping_frequency[env_ids] = self._stepping_frequency_base
#         self._swing_cutoff[env_ids] = 2 * torch.pi * (1 - self._swing_ratio)
#         self._prev_frame_robot_time[env_ids] = self._robot.time_since_reset[env_ids]
#         self._first_stance_seen[env_ids] = 0
#     
#     def update(self):
#         current_robot_time = self._robot.time_since_reset
#         delta_t = current_robot_time - self._prev_frame_robot_time
#         self._prev_frame_robot_time = current_robot_time.clone()
#         self._current_phase += 2 * torch.pi * self._stepping_frequency[:, None] * delta_t[:, None]
#     
#     @property
#     def desired_contact_state(self):
#         modulated_phase = torch.remainder(self._current_phase + 2 * torch.pi, 2 * torch.pi)
#         raw_contact = torch.where(modulated_phase > self._swing_cutoff, False, True)
#         self._first_stance_seen = torch.logical_or(self._first_stance_seen, raw_contact)
#         return torch.where(self._first_stance_seen, raw_contact, torch.ones_like(raw_contact))
#     
#     @property
#     def normalized_phase(self):
#         modulated_phase = torch.remainder(self._current_phase + 2 * torch.pi, 2 * torch.pi)
#         return torch.where(
#             modulated_phase < self._swing_cutoff,
#             modulated_phase / self._swing_cutoff,
#             (modulated_phase - self._swing_cutoff) / (2 * torch.pi - self._swing_cutoff)
#         )
#     
#     @property
#     def stance_duration(self):
#         return self._swing_cutoff / (2 * torch.pi * self._stepping_frequency[:, None])
#     
#     @property
#     def stepping_frequency(self):
#         return self._stepping_frequency
#     
#     @stepping_frequency.setter
#     def stepping_frequency(self, new_frequency):
#         self._stepping_frequency = new_frequency
#
#
# # ============================================================================
# # SIMULATION SETUP
# # ============================================================================
# def create_sim(sim_conf):
#     gym = gymapi.acquire_gym()
#     _, sim_device_id = gymutil.parse_device_str(sim_conf.sim_device)
#     graphics_device_id = sim_device_id if sim_conf.show_gui else -1
#     sim = gym.create_sim(sim_device_id, graphics_device_id,
#                          sim_conf.physics_engine, sim_conf.sim_params)
#     viewer = None
#     if sim_conf.show_gui:
#         viewer = gym.create_viewer(sim, gymapi.CameraProperties())
#         gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "QUIT")
#         gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "PAUSE")
#         # Velocity controls
#         gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "CURRENT_VEL_UP")
#         gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "CURRENT_VEL_DOWN")
#         gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "DESIRED_VEL_UP")
#         gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "DESIRED_VEL_DOWN")
#         # Frequency controls
#         gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP, "FREQ_UP")
#         gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN, "FREQ_DOWN")
#     plane_params = gymapi.PlaneParams()
#     plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
#     plane_params.static_friction = 1.0
#     plane_params.dynamic_friction = 1.0
#     gym.add_ground(sim, plane_params)
#     return gym, sim, viewer
#
#
# def get_init_positions(num_envs, device):
#     init_positions = torch.zeros((num_envs, 3), device=device)
#     init_positions[:, 2] = 0.34
#     return init_positions
#
#
# def main(argv):
#     del argv
#     
#     sim_conf = sim_config.get_config(use_gpu=FLAGS.use_gpu, show_gui=FLAGS.show_gui)
#     gym, sim, viewer = create_sim(sim_conf)
#     
#     robot = go1.Go1(
#         num_envs=FLAGS.num_envs,
#         init_positions=get_init_positions(FLAGS.num_envs, sim_conf.sim_device),
#         sim=sim,
#         viewer=viewer,
#         sim_config=sim_conf,
#         motor_control_mode=MotorControlMode.POSITION
#     )
#     
#     # Use embedded PhaseGaitGenerator
#     gait_generator = PhaseGaitGenerator(
#         robot=robot,
#         initial_offset=GAIT_INITIAL_OFFSET,       # CONFIG: Phase offsets
#         swing_ratio=GAIT_SWING_RATIO,             # CONFIG: Swing ratio
#         stepping_frequency=GAIT_STEPPING_FREQUENCY # CONFIG: Stepping frequency
#     )
#     
#     # Use the actual RaibertSwingLegController from your project
#     swing_leg_controller = raibert_swing_leg_controller.RaibertSwingLegController(
#         robot, gait_generator,
#         foot_landing_clearance=SWING_FOOT_LANDING_CLEARANCE,  # CONFIG
#         foot_height=SWING_FOOT_HEIGHT                          # CONFIG
#     )
#     
#     robot.reset()
#     
#     dt = sim_conf.dt * sim_conf.action_repeat
#     desired_velocity = torch.zeros((FLAGS.num_envs, 3), device=robot.device)
#     env_ids = torch.arange(FLAGS.num_envs, device=robot.device, dtype=torch.int32)
#     
#     # Track stance foot positions
#     stance_foot_world_positions = torch.zeros((FLAGS.num_envs, 4, 3), device=robot.device)
#     prev_contact_state = torch.zeros((FLAGS.num_envs, 4), dtype=torch.bool, device=robot.device)
#     
#     # Fake velocity controls (what we tell the Raibert controller)
#     fake_current_vx = INITIAL_FAKE_CURRENT_VX
#     fake_desired_vx = INITIAL_DESIRED_VX
#     current_frequency = GAIT_STEPPING_FREQUENCY
#     paused = False
#     
#     print("=" * 70)
#     print("Phase Gait Visualizer - VELOCITY FAKER MODE")
#     print("=" * 70)
#     print(f"Gait frequency: {GAIT_STEPPING_FREQUENCY} Hz")
#     print(f"Swing ratio: {GAIT_SWING_RATIO}")
#     print(f"Phase offsets: {GAIT_INITIAL_OFFSET}")
#     print("=" * 70)
#     print("KEYBOARD CONTROLS:")
#     print("  W/S  - Increase/Decrease FAKE current velocity")
#     print("  D/A  - Increase/Decrease desired velocity")
#     print("  UP/DOWN - Increase/Decrease stepping frequency")
#     print("  SPACE - Pause/Resume")
#     print("  ESC - Quit")
#     print("=" * 70)
#     print(f"Initial: Current V = {fake_current_vx:.2f} m/s, Desired V = {fake_desired_vx:.2f} m/s")
#     print("=" * 70)
#     
#     pbar = tqdm(total=FLAGS.total_time_secs)
#     steps = 0
#     
#     with torch.inference_mode():
#         while robot.time_since_reset[0] <= FLAGS.total_time_secs:
#             # Poll keyboard events
#             if FLAGS.show_gui:
#                 for evt in gym.query_viewer_action_events(viewer):
#                     if evt.action == "QUIT" and evt.value > 0:
#                         print("\nQuit requested")
#                         pbar.close()
#                         return
#                     if evt.action == "PAUSE" and evt.value > 0:
#                         paused = not paused
#                         print(f"\n{'PAUSED' if paused else 'RESUMED'}")
#                     if evt.action == "CURRENT_VEL_UP" and evt.value > 0:
#                         fake_current_vx = min(fake_current_vx + VELOCITY_INCREMENT, MAX_VELOCITY)
#                         print(f"\nCurrent V: {fake_current_vx:.2f} m/s | Desired V: {fake_desired_vx:.2f} m/s | Freq: {current_frequency:.2f} Hz")
#                     if evt.action == "CURRENT_VEL_DOWN" and evt.value > 0:
#                         fake_current_vx = max(fake_current_vx - VELOCITY_INCREMENT, MIN_VELOCITY)
#                         print(f"\nCurrent V: {fake_current_vx:.2f} m/s | Desired V: {fake_desired_vx:.2f} m/s | Freq: {current_frequency:.2f} Hz")
#                     if evt.action == "DESIRED_VEL_UP" and evt.value > 0:
#                         fake_desired_vx = min(fake_desired_vx + VELOCITY_INCREMENT, MAX_VELOCITY)
#                         print(f"\nCurrent V: {fake_current_vx:.2f} m/s | Desired V: {fake_desired_vx:.2f} m/s | Freq: {current_frequency:.2f} Hz")
#                     if evt.action == "DESIRED_VEL_DOWN" and evt.value > 0:
#                         fake_desired_vx = max(fake_desired_vx - VELOCITY_INCREMENT, MIN_VELOCITY)
#                         print(f"\nCurrent V: {fake_current_vx:.2f} m/s | Desired V: {fake_desired_vx:.2f} m/s | Freq: {current_frequency:.2f} Hz")
#                     if evt.action == "FREQ_UP" and evt.value > 0:
#                         current_frequency = min(current_frequency + 0.1, 4.0)
#                         gait_generator._stepping_frequency[:] = current_frequency
#                         gait_generator._swing_cutoff[:] = 2 * torch.pi * (1 - gait_generator._swing_ratio)
#                         print(f"\nCurrent V: {fake_current_vx:.2f} m/s | Desired V: {fake_desired_vx:.2f} m/s | Freq: {current_frequency:.2f} Hz")
#                     if evt.action == "FREQ_DOWN" and evt.value > 0:
#                         current_frequency = max(current_frequency - 0.1, 0.5)
#                         gait_generator._stepping_frequency[:] = current_frequency
#                         gait_generator._swing_cutoff[:] = 2 * torch.pi * (1 - gait_generator._swing_ratio)
#                         print(f"\nCurrent V: {fake_current_vx:.2f} m/s | Desired V: {fake_desired_vx:.2f} m/s | Freq: {current_frequency:.2f} Hz")
#             
#             if paused:
#                 if FLAGS.show_gui:
#                     robot.render()
#                 continue
#             
#             t = robot.time_since_reset[0].item()
#             
#             # Set the desired velocity for the swing leg controller
#             desired_velocity[:, 0] = fake_desired_vx
#             
#             # ===== KINEMATIC MODE: Direct joint position control =====
#             
#             # Fix base position (robot stays stationary, but we fake velocity)
#             robot._root_states[:, 0] = 0.0
#             robot._root_states[:, 1] = 0.0
#             robot._root_states[:, 2] = FIXED_HEIGHT
#             robot._root_states[:, 3:7] = torch.tensor([0., 0., 0., 1.], device=robot.device)
#             robot._root_states[:, 7] = 0.0  # Actual velocity is 0 (stationary)
#             robot._root_states[:, 8:13] = 0.0
#             gym.set_actor_root_state_tensor_indexed(
#                 sim, gymtorch.unwrap_tensor(robot._root_states),
#                 gymtorch.unwrap_tensor(env_ids), len(env_ids))
#             gym.refresh_actor_root_state_tensor(sim)
#             robot._post_physics_step()
#             
#             gait_generator.update()
#             
#             # ===== FAKE THE VELOCITY FOR RAIBERT CONTROLLER =====
#             # Save the real velocity
#             real_base_velocity_world = robot.base_vel.clone()
#             
#             # Set fake velocity (this is what Raibert controller will see)
#             robot.base_vel[:, 0] = fake_current_vx
#             robot.base_vel[:, 1] = 0.0
#             robot.base_vel[:, 2] = 0.0
#             
#             # Also need to fake base_velocity_world_frame if it's a separate property
#             # The Raibert controller uses self._robot.base_velocity_world_frame
#             real_base_velocity_world_frame = robot.base_velocity_world_frame.clone()
#             
#             # Update swing leg controller with faked velocity
#             swing_leg_controller.update(desired_velocity)
#             
#             # Restore real velocity
#             robot.base_vel[:] = real_base_velocity_world
#             
#             # ===== END VELOCITY FAKING =====
#             
#             contact_state = gait_generator.desired_contact_state
#             
#             # Detect touchdown
#             just_touched_down = contact_state & ~prev_contact_state
#             current_foot_world = robot.foot_positions_in_world_frame
#             for leg_idx in range(4):
#                 mask = just_touched_down[:, leg_idx]
#                 if mask.any():
#                     stance_foot_world_positions[mask, leg_idx] = current_foot_world[mask, leg_idx]
#             
#             # Swing feet from Raibert controller
#             swing_foot_positions_world = swing_leg_controller.desired_foot_positions
#             
#             # Convert to base frame
#             base_pos = robot._root_states[:, :3]
#             
#             stance_foot_relative = stance_foot_world_positions - base_pos[:, None, :]
#             stance_foot_base = torch.bmm(
#                 robot.base_rot_mat_t,
#                 stance_foot_relative.transpose(1, 2)
#             ).transpose(1, 2)
#             
#             swing_foot_base = torch.bmm(
#                 robot.base_rot_mat_t,
#                 swing_foot_positions_world.transpose(1, 2)
#             ).transpose(1, 2)
#             
#             target_foot_base = torch.where(
#                 contact_state[:, :, None].expand(-1, -1, 3),
#                 stance_foot_base,
#                 swing_foot_base
#             )
#             
#             target_foot_base[:, :, 2] = torch.clip(target_foot_base[:, :, 2], min=-0.35, max=-0.05)
#             
#             joint_positions = robot.get_motor_angles_from_foot_positions(target_foot_base)
#             
#             # Directly set joint positions
#             robot._dof_state.view(FLAGS.num_envs, 12, 2)[..., 0] = joint_positions
#             robot._dof_state.view(FLAGS.num_envs, 12, 2)[..., 1] = 0.0
#             gym.set_dof_state_tensor_indexed(
#                 sim, gymtorch.unwrap_tensor(robot._dof_state),
#                 gymtorch.unwrap_tensor(env_ids), len(env_ids))
#             
#             gym.simulate(sim)
#             gym.refresh_dof_state_tensor(sim)
#             gym.refresh_actor_root_state_tensor(sim)
#             gym.refresh_rigid_body_state_tensor(sim)
#             robot._post_physics_step()
#             
#             prev_contact_state = contact_state.clone()
#             robot._time_since_reset += dt
#             
#             steps += 1
#             # pbar.update(dt)
#             
#             if steps % 100 == 0:
#                 contact = contact_state[0].cpu().numpy().astype(int)
#                 phase = gait_generator.normalized_phase[0].cpu().numpy()
#                 delta_v = fake_current_vx - fake_desired_vx
#                 print(f"t={t:.2f}s | V_curr={fake_current_vx:.2f} | V_des={fake_desired_vx:.2f} | ΔV={delta_v:.2f} | freq={current_frequency:.2f} | contact={contact}")
#             
#             if FLAGS.show_gui:
#                 # Draw foot position spheres
#                 current_foot_world = robot.foot_positions_in_world_frame
#                 draw_foot_spheres(gym, viewer, current_foot_world, contact_state)
#                 robot.render()
#     
#     pbar.close()
#     print("Done.")
#
#
# if __name__ == "__main__":
#     app.run(main)
