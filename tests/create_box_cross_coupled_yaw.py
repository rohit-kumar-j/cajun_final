import numpy as np
from typing import List, Optional, Tuple
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from loguru import logger
import sys
import torch
import math

def quaternion_inverse(q):
    """Get inverse of quaternion [x, y, z, w]"""
    x, y, z, w = q
    return [-x, -y, -z, w]

def quaternion_multiply(q1, q2):
    """Multiply two quaternions"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ]

def project_quaternion_to_yaw(q):
    """Project quaternion to yaw-only rotation (around Z-axis)"""
    x, y, z, w = q
    # Extract yaw and create yaw-only quaternion
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return [0, 0, math.sin(yaw/2), math.cos(yaw/2)]

# BEFORE the loop - initialize
prev_orientation = [0.0, 0.0, 0.0, 1.0]  # Identity quaternion
camera_yaw_quat = [0.0, 0.0, 0.0, 1.0]  # Camera's yaw tracking



def quaternion_to_forward_vector(quat):
    """Get forward direction vector from quaternion (direction actor is facing)."""
    x, y, z, w = quat
    
    # Forward vector (assuming +X is forward in local space)
    # Rotate the unit X vector [1, 0, 0] by the quaternion
    forward_x = 1 - 2*(y*y + z*z)
    forward_y = 2*(x*y + w*z)
    forward_z = 2*(x*z - w*y)
    
    return forward_x, forward_y, forward_z

def quaternion_to_yaw(quat):
    """Extract yaw angle from quaternion [x, y, z, w]."""
    x, y, z, w = quat
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw

def increment_angles(quat, rpy):
    """
    Increment roll, pitch, yaw by multiplying with delta quaternions.
    Avoids gimbal lock by staying in quaternion space.
    
    Args:
        quat: Current quaternion [x, y, z, w]
        rpy: [roll, pitch, yaw] increments in radians
    
    Returns:
        New quaternion [x, y, z, w]
    """
    delta_roll, delta_pitch, delta_yaw = rpy
    
    # Create delta quaternions for each axis
    # Roll (X-axis)
    qr = [
        math.sin(delta_roll * 0.5),
        0.0,
        0.0,
        math.cos(delta_roll * 0.5)
    ]
    
    # Pitch (Y-axis)
    qp = [
        0.0,
        math.sin(delta_pitch * 0.5),
        0.0,
        math.cos(delta_pitch * 0.5)
    ]
    
    # Yaw (Z-axis)
    qy = [
        0.0,
        0.0,
        math.sin(delta_yaw * 0.5),
        math.cos(delta_yaw * 0.5)
    ]
    
    # Quaternion multiplication helper
    def quat_multiply(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return [
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ]
    
    # Apply rotations: current * roll * pitch * yaw
    result = quat_multiply(quat, qr)
    result = quat_multiply(result, qp)
    result = quat_multiply(result, qy)
    
    return result

logger.remove()
logger.add(sys.stderr, 
           format="[{time}] | [<level>{level: <8}</level>] | <level>{message}</level>", # | <level>{extra}</level>",     
           colorize=True,
           level="TRACE"
           )
# logger.add("app.log", format="{time:HH:MM:SS} | {level: <8} | {message}")

# Initialize Isaac Gym
gym = gymapi.acquire_gym()

# Parse arguments
args = gymutil.parse_arguments(description="Collision-free box demo")


class SimHandler():

    def __init__(self):
        # Logging (simplified - using module-level config)
        self.logger = logger
        self.logger.trace("SimHandler initialized")
        self.logger.success("SimHandler initialized")
        self.logger.info("SimHandler initialized")
        self.logger.debug("SimHandler initialized")
        self.logger.error("SimHandler initialized")
        self.logger.warning("SimHandler initialized")
        self.logger.critical("SimHandler initialized")
        
        # Initialize attributes
        self.sim = None
        self.env = None
        self.box_assets = []
        self.sphere_assets = []
        self.actors = []
        self.viewer = None
        self.actor = None
        self.device = "cuda" if True and torch.cuda.is_available() else "cpu"
        # self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.logger.info(f"device = {self.device}")

    def create_sim(self, compute_device=0, graphics_device=0, use_gpu=True):
        self.logger.info("Creating sim")
        
        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim_params.dt = 1.0 / 60.0
        self.sim_params.substeps = 2
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 6
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.use_gpu = use_gpu
        
        # FIXED: Use gymapi.SIM_PHYSX and self.sim_params
        self.sim = gym.create_sim(
            compute_device, 
            graphics_device, 
            gymapi.SIM_PHYSX,
            self.sim_params
        )
        return self.sim

    def create_env(self):
        self.logger.info("Creating env")
        
        self.env_lb = gymapi.Vec3(-1.0, -1.0, 0.0)
        self.env_ub = gymapi.Vec3(1.0, 1.0, 1.0)
        self.n_envs_in_row = 1
        self.env = gym.create_env(self.sim, self.env_lb, self.env_ub, self.n_envs_in_row)
        return self.env

    def create_box(self, size: Optional[List[float]] = None, name: Optional[str] = None) -> gymapi.Asset:
        if size is None:  # FIXED: == instead of =
            size = [0.5, 0.5, 0.5]
        
        if len(size) != 3:
            self.logger.error("size must have 3 elements. ex: create_box(size=[0.5,0.5,0.5])")
            return None

        asset_opts = gymapi.AssetOptions()
        asset_opts.fix_base_link = True
        asset_opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_opts.override_inertia = False
        asset_opts.override_com = False

        w, h, d = size
        box_asset = gym.create_box(self.sim, w, h, d, asset_opts)  # FIXED: self.sim
        self.box_assets.append(box_asset)
        return box_asset

    def create_sphere(self, radius: float, name: Optional[str] = None) -> gymapi.Asset:
        if radius is None:
            radius = 0.5
            self.logger.error("No radius provided, defaluting to radius: .5")

        asset_opts = gymapi.AssetOptions()
        asset_opts.fix_base_link = True
        asset_opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_opts.override_inertia = False
        asset_opts.override_com = False

        sphere_asset = gym.create_sphere(self.sim, radius, asset_opts)  # FIXED: self.sim
        self.sphere_assets.append(sphere_asset)
        return sphere_asset

    def add_ground(self, z_height: float = 0.0):  # FIXED: Added self, default value
        self.logger.info(f"Adding ground plane at z={z_height}")

        self.plane_params = gymapi.PlaneParams()
        self.plane_params.normal = gymapi.Vec3(0, 0, 1)  # FIXED: Should be (0,0,1) for up
        self.plane_params.distance = z_height
        gym.add_ground(self.sim, self.plane_params)

    def set_pose(self, actor, p: List[float] = None, r: List[float] = None):
        """Move an existing actor to a new pose."""
        if p is None:
            p = [0, 0, 0]
        if r is None:
            r = [0, 0, 0, 1]
        
        if actor not in self.actors:
            self.logger.error(f"Actor(idx={actor}) not found in registry")
            return
        actor_idx = self.actors[actor]
        
        # Root state format: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        self.root_states[actor_idx, 0:3] = torch.tensor(p, device=self.device)
        self.root_states[actor_idx, 3:7] = torch.tensor(r, device=self.device)
        
        # Optional: set velocities to zero
        self.root_states[actor_idx, 7:13] = 0

        gym.set_actor_root_state_tensor(self.sim, self.root_tensor)
        self.logger.debug(f"Actor moved to p={p}, r={r}")

    def get_pose(self, actor) -> Tuple[List[float], List[float]]:
        """Get current pose of an actor. Returns (position, rotation)."""
        if actor not in self.actors:
            self.logger.error(f"Actor(idx={actor}) not found in registry")
            return None, None
        
        actor_idx = self.actors[actor]
        
        # Extract position and rotation from root state tensor
        # Root state format: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        pos = self.root_states[actor_idx, 0:3].cpu().tolist()
        rot = self.root_states[actor_idx, 3:7].cpu().tolist()
        
        self.logger.debug(f"Actor pose: p={pos}, r={rot}")
        return pos, rot

    def transform(self, asset: gymapi.Asset, p: Optional[List[float]] = None, r: Optional[List[float]] = None, name: Optional[str] = None):
        """This is done to first place an actor in the scene"""
        if p is None: p = [0.0, 0.0, 0.5]
        if r is None: r = [0.0, 0.0, 0.0, 1.0]  # Identity quaternion
        
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(p[0], p[1], p[2])
        pose.r = gymapi.Quat(r[0], r[1], r[2], r[3])  # FIXED: Proper quaternion

        if name is None:
            actor_num =  len(self.actors) + 1
            name = f"actor_{actor_num}"
        actor = gym.create_actor(
            self.env,
            asset, 
            pose, 
            name, 
            -1,  # collision group
            -1,  # collision filter
            0    # segmentation id
        )
        self.actors.append(actor)
        self.logger.success(f"Actor '{name}' created at {p}")
        return actor

    def create_viewer(self):
        self.viewer = gym.create_viewer(self.sim, gymapi.CameraProperties())
        return self.viewer

    def prepare_sim(self):
        """Call AFTER creating actors, BEFORE simulation loop."""
        gym.prepare_sim(self.sim)
        
        # Acquire root state tensor
        self.root_tensor = gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.root_tensor)
        
        self.logger.info("Simulation prepared with tensor access")


if  __name__ == "__main__":
    with  logger.catch():
        # Main execution
        sim_handler = SimHandler()
        sim = sim_handler.create_sim(args.compute_device_id, args.graphics_device_id, args.use_gpu)
        if sim is None:
            sim_handler.logger.error("-----> Failed to create sim")
            quit()

        sim_handler.create_env()
        b1 = sim_handler.create_box([0.1, 0.3, 0.7])
        actor1 = sim_handler.transform(b1,p=[1.0, 0.0, 0.5])
        gym.set_rigid_body_color(
            sim_handler.env,      # environment handle
            actor1,               # actor handle (not asset!)
            0,                    # rigid body index
            gymapi.MESH_VISUAL,   # mesh type
            gymapi.Vec3(0., 1., 0.)  # RGB color as Vec3
        )


        b2 = sim_handler.create_box()
        actor2 = sim_handler.transform(b1,p=[1.0, 0.0, 0.5])
        gym.set_rigid_body_color(
            sim_handler.env,      # environment handle
            actor1,               # actor handle (not asset!)
            0,                    # rigid body index
            gymapi.MESH_VISUAL,   # mesh type
            gymapi.Vec3(0., 1., 1.)  # RGB color as Vec3
        )

        s1 = sim_handler.create_sphere(0.1)
        actor3 = sim_handler.transform(s1,p=[1.0, 0.0, 0.5])
        gym.set_rigid_body_color(
            sim_handler.env,      # environment handle
            actor3,               # actor handle (not asset!)
            0,                    # rigid body index
            gymapi.MESH_VISUAL,   # mesh type
            gymapi.Vec3(1., 0., 1.)  # RGB color as Vec3
        )

        sim_handler.add_ground(0)


        sim_handler.prepare_sim()

        sim_handler.set_pose(actor1,p=[2.0, 0.0, 0.5])
        sim_handler.set_pose(actor2,p=[1.0, 2.0, 0.5])

        viewer = sim_handler.create_viewer()
        if viewer is None:
            sim_handler.logger.error("-----> Failed to create viewer")
            quit()

        tracked_yaw = 0.0

        # Main simulation loop
        while not gym.query_viewer_has_closed(viewer):
            # update sim_time
            sim_time = torch.Tensor([gym.get_sim_time(sim_handler.sim)])
            logger.trace(f"simtime: {sim_time}")

            offset_x = 0.01*torch.cos(sim_time)
            offset_y = 0.01*torch.sin(sim_time)

            p1, r1 = sim_handler.get_pose(actor1)
            sim_handler.set_pose(actor1,p=[p1[0] + offset_x, p1[1] + offset_y, p1[2]])

            p2, r2 = sim_handler.get_pose(actor3)
            sim_handler.set_pose(actor3,p=[p2[0] + offset_x, p2[1] - offset_y, p2[2]])

            p3, r3 = sim_handler.get_pose(actor2)
            new_r3 = increment_angles(r3, [0.1, 0.01,0.])  # Rotate 0.01 radians per frame
            sim_handler.set_pose(actor2,p=[p3[0] - offset_x, p3[1] + offset_y, p3[2]],r=new_r3)


            # Compute delta rotation: delta = new * inverse(prev)
            prev_inv = quaternion_inverse(prev_orientation)
            delta_rotation = quaternion_multiply(new_r3, prev_inv)
            
            # Project delta to yaw-only component
            delta_yaw_only = project_quaternion_to_yaw(delta_rotation)
            
            # Apply yaw delta to camera orientation
            camera_yaw_quat = quaternion_multiply(camera_yaw_quat, delta_yaw_only)
            
            # Update previous orientation
            prev_orientation = new_r3
            
            # Extract yaw angle from camera quaternion
            yaw = math.atan2(2 * (camera_yaw_quat[3] * camera_yaw_quat[2] + camera_yaw_quat[0] * camera_yaw_quat[1]), 
                             1 - 2 * (camera_yaw_quat[1]**2 + camera_yaw_quat[2]**2))
            
            # Camera parameters
            cam_distance = 2.0
            cam_height = 0.5

            # Position camera based on tracked yaw
            cam_pos = gymapi.Vec3(
                p3[0] - cam_distance * math.cos(yaw),
                p3[1] - cam_distance * math.sin(yaw),
                p3[2] + cam_height
            )

            target_pos = gymapi.Vec3(p3[0], p3[1], p3[2])
            gym.viewer_camera_look_at(viewer, None, cam_pos, target_pos)

            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

        # Cleanup
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)
        logger.info("Exiting sim")

