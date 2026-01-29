"""Visualizer for gait configurations"""
import os
import sys
import importlib

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
from src.controllers import phase_gait_generator
from src.controllers import qp_torque_optimizer
from src.controllers import raibert_swing_leg_controller
from src.robots import go1
from src.robots.motors import MotorControlMode

flags.DEFINE_string("config", "g2_rotary", "Config name from src/envs/configs/")
flags.DEFINE_integer("num_envs", 1, "Number of environments.")
flags.DEFINE_float("total_time_secs", 20., "Total time.")
flags.DEFINE_bool("use_gpu", True, "Use GPU.")
flags.DEFINE_bool("show_gui", True, "Show GUI.")
flags.DEFINE_bool("fixed_base", True, "True: kinematic (direct joint pos), False: dynamic (motor torques)")
FLAGS = flags.FLAGS


def load_config(config_name):
    module = importlib.import_module(f"src.envs.configs.{config_name}")
    return module.get_config()


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
    
    config = load_config(FLAGS.config)
    
    with config.unlocked():
        config.swing_foot_height = config.get('swing_foot_height', 0.1) + 0.15
        config.desired_vx = config.get('velocity_ub', 3.0)
        config.ramp_time = 10.0
        config.fixed_height = 0.26  # Standard Go1 standing height
        
        # Dynamic mode
        if not FLAGS.fixed_base:
            config.desired_vx = min(config.desired_vx, 1.0)  # Cap at 1 m/s for stability

    sim_conf = sim_config.get_config(use_gpu=FLAGS.use_gpu, show_gui=FLAGS.show_gui)
    gym, sim, viewer = create_sim(sim_conf)
    
    # Use HYBRID mode for torque control, POSITION for kinematic
    motor_mode = MotorControlMode.POSITION if FLAGS.fixed_base else MotorControlMode.HYBRID
    
    robot = go1.Go1(
        num_envs=FLAGS.num_envs,
        init_positions=get_init_positions(FLAGS.num_envs, sim_conf.sim_device),
        sim=sim,
        viewer=viewer,
        sim_config=sim_conf,
        motor_control_mode=motor_mode
    )
    
    gait_generator = phase_gait_generator.PhaseGaitGenerator(robot, config.gait)
    swing_leg_controller = raibert_swing_leg_controller.RaibertSwingLegController(
        robot, gait_generator,
        foot_landing_clearance=config.swing_foot_landing_clearance,
        foot_height=config.swing_foot_height
    )
    torque_optimizer = qp_torque_optimizer.QPTorqueOptimizer(
        robot,
        desired_body_height=config.fixed_height,
        base_position_kp=config.base_position_kp,
        base_position_kd=config.base_position_kd,
        base_orientation_kp=config.base_orientation_kp,
        base_orientation_kd=config.base_orientation_kd,
        weight_ddq=config.qp_weight_ddq,
        foot_friction_coef=config.qp_foot_friction_coef,
        body_inertia=config.qp_body_inertia,
        use_full_qp=config.use_full_qp,
        clip_grf=config.clip_grf_in_sim
    )
    
    robot.reset()
    
    dt = sim_conf.dt * sim_conf.action_repeat
    desired_velocity = torch.zeros((FLAGS.num_envs, 3), device=robot.device)
    env_ids = torch.arange(FLAGS.num_envs, device=robot.device, dtype=torch.int32)
    accumulated_x = 0.0
    
    # For fixed_base mode: track stance foot positions
    stance_foot_world_positions = torch.zeros((FLAGS.num_envs, 4, 3), device=robot.device)
    prev_contact_state = torch.zeros((FLAGS.num_envs, 4), dtype=torch.bool, device=robot.device)
    
    print(f"Config: {FLAGS.config}")
    print(f"Mode: {'Fixed base (kinematic)' if FLAGS.fixed_base else 'Free base (dynamic)'}")
    print(f"Target velocity: {config.desired_vx} m/s")
    print(f"Swing foot height: {config.swing_foot_height}")
    
    pbar = tqdm(total=FLAGS.total_time_secs)
    steps = 0
    
    with torch.inference_mode():
        while robot.time_since_reset[0] <= FLAGS.total_time_secs:
            t = robot.time_since_reset[0].item()
            
            vx = config.desired_vx * min(t / config.ramp_time, 1.0)
            accumulated_x += vx * dt
            
            desired_velocity[:, 0] = vx
            
            if FLAGS.fixed_base:
                # ===== KINEMATIC MODE: Direct joint position control =====
                
                # Fix base position
                robot._root_states[:, 0] = 0.0
                robot._root_states[:, 1] = 0.0
                robot._root_states[:, 2] = config.fixed_height
                robot._root_states[:, 3:7] = torch.tensor([0., 0., 0., 1.], device=robot.device)
                robot._root_states[:, 7] = vx
                robot._root_states[:, 8:13] = 0.0
                gym.set_actor_root_state_tensor_indexed(
                    sim, gymtorch.unwrap_tensor(robot._root_states),
                    gymtorch.unwrap_tensor(env_ids), len(env_ids))
                gym.refresh_actor_root_state_tensor(sim)
                robot._post_physics_step()
                
                gait_generator.update()
                swing_leg_controller.update(desired_velocity)
                
                contact_state = gait_generator.desired_contact_state
                
                # Detect touchdown
                just_touched_down = contact_state & ~prev_contact_state
                current_foot_world = robot.foot_positions_in_world_frame
                for leg_idx in range(4):
                    mask = just_touched_down[:, leg_idx]
                    if mask.any():
                        stance_foot_world_positions[mask, leg_idx] = current_foot_world[mask, leg_idx]
                
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
                
            else:
                # ===== DYNAMIC MODE: Motor torques via QP optimizer =====
                # Similar to JumpEnv.step() inner loop
                
                gait_generator.update()
                swing_leg_controller.update(desired_velocity)
                
                # Set desired base position: track current x,y, maintain height
                torque_optimizer.desired_base_position = torch.stack(
                    (robot.base_position[:, 0],
                     robot.base_position[:, 1],
                     torch.full((FLAGS.num_envs,), config.fixed_height, device=robot.device)),
                    dim=1)
                
                # Set desired velocities
                torque_optimizer.desired_linear_velocity = torch.stack(
                    (desired_velocity[:, 0],
                     torch.zeros(FLAGS.num_envs, device=robot.device),
                     torch.zeros(FLAGS.num_envs, device=robot.device)),
                    dim=1)
                
                # Desired orientation: level (no roll/pitch), maintain current yaw
                torque_optimizer.desired_base_orientation_rpy = torch.stack(
                    (torch.zeros(FLAGS.num_envs, device=robot.device),
                     torch.zeros(FLAGS.num_envs, device=robot.device),
                     robot.base_orientation_rpy[:, 2]),
                    dim=1)
                
                # Zero angular velocity
                torque_optimizer.desired_angular_velocity = torch.zeros(
                    (FLAGS.num_envs, 3), device=robot.device)
                
                # Get motor action from QP torque optimizer
                motor_action, _, _, _, _ = torque_optimizer.get_action(
                    gait_generator.desired_contact_state,
                    swing_foot_position=swing_leg_controller.desired_foot_positions
                )
                
                # Step robot with motor action (this applies torques)
                robot.step(motor_action)
            
            steps += 1
            pbar.update(dt)
            
            if steps % 100 == 0:
                contact = gait_generator.desired_contact_state[0].cpu().numpy()
                base_pos = robot.base_position[0].cpu().numpy()
                print(f"t={t:.2f}s vx={vx:.2f} pos={base_pos} contact={contact}")
            
            if FLAGS.show_gui:
                robot.render()
    
    pbar.close()
    print(f"Done.")


if __name__ == "__main__":
    app.run(main)

