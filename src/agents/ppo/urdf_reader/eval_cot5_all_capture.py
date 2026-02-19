"""Evaluate a trained policy and save detailed data for MATLAB analysis."""
from absl import app
from absl import flags

from datetime import datetime
import os
import signal
import sys
import time
import shutil
import subprocess

from isaacgym import gymapi
from isaacgym.torch_utils import to_torch
from rsl_rl.runners import OnPolicyRunner
import torch
import yaml
import numpy as np
from tqdm import tqdm


_exit_state = {
    'data_arrays': None,
    'data_count': 0,
    'output_dir': None,
    'steps_count': 0,
    'start_time': None,
    'data_saved': False,
    'gait_name': 'unknown',
    'frames_dir': None,
    'video_path': None,
}

_frame_count = 0
_next_frame_time = 0.0
_frame_interval = 0.0
_frame_template = ""
_gym = None
_viewer = None
_env = None


def draw_box(gym, viewer, env, center=None, corners=None, color=[1, 0, 0], thickness=3):
    if center is not None:
        if len(center) == 5:
            x, y, z, half_length, half_width = center
        elif len(center) == 4:
            x, y, half_length, half_width = center
            z = 0.01
        else:
            raise ValueError("center must be length 4 or 5")
        corners = [
            [x - half_length, y - half_width, z],
            [x + half_length, y - half_width, z],
            [x + half_length, y + half_width, z],
            [x - half_length, y + half_width, z]
        ]
    elif corners is not None:
        corners = [list(c) + [0.01] * (3 - len(c)) for c in corners]
        if len(corners) != 4:
            raise ValueError("Must provide exactly 4 corner points")
    else:
        raise ValueError("Must provide either 'center' or 'corners'")

    lines = []
    offset_step = 0.002
    for t in range(thickness):
        z_offset = t * offset_step
        for i in range(4):
            start = corners[i].copy(); end = corners[(i + 1) % 4].copy()
            start[2] += z_offset; end[2] += z_offset
            lines.append(start + end)
    for t in range(max(1, thickness // 2)):
        z_offset = t * offset_step
        d1s = corners[0].copy(); d1e = corners[2].copy()
        d1s[2] += z_offset; d1e[2] += z_offset; lines.append(d1s + d1e)
        d2s = corners[1].copy(); d2e = corners[3].copy()
        d2s[2] += z_offset; d2e[2] += z_offset; lines.append(d2s + d2e)

    lines = np.array(lines, dtype=np.float32)
    colors_array = np.array([color] * len(lines), dtype=np.float32)
    gym.add_lines(viewer, env._robot._envs[0], lines.shape[0], lines, colors_array)


def capture_frame(sim_time):
    global _frame_count, _next_frame_time
    if sim_time >= _next_frame_time:
        if FLAGS.render_landing_pos and _env is not None:
            _gym.clear_lines(_viewer)
            env_id = 0
            robot_pos = _env._robot.base_position[env_id].cpu().numpy()
            desired_foot_positions = _env._swing_leg_controller.desired_foot_positions
            colors = [[1,0.2,0.2],[0.2,1,0.2],[0.3,0.3,1],[1,1,0.2]]
            for foot_id in range(4):
                fp = desired_foot_positions[env_id, foot_id].cpu().numpy()
                draw_box(_gym, _viewer, _env,
                         center=(fp[0]+robot_pos[0], fp[1]+robot_pos[1], 0.02, 0.05, 0.05),
                         color=colors[foot_id], thickness=5)
        frame_path = _frame_template.format(_frame_count)
        _gym.write_viewer_image_to_file(_viewer, frame_path)
        _frame_count += 1
        _next_frame_time += _frame_interval


def save_data_to_files(output_dir, data_arrays, data_count):
    """
    Save all data to individual .txt files — ONE FLOAT PER LINE.

    File layout
    -----------
    Torso:
      PosTorso0.txt  torso_x (world)
      PosTorso1.txt  torso_pitch
      PosTorso2.txt  torso_roll
      PosTorso3.txt  torso_yaw
      PosTorso4.txt  torso_y (world)
      PosTorso5.txt  torso_z (world)
      PosTorso6-8    vx, vy, vz (world)

    Joints (i = 0..11, order FR_hip/thigh/calf, FL, RR, RL):
      q<i>.txt        joint position  (rad)
      dq<i>.txt       joint velocity  (rad/s)
      tauM<i>.txt     joint torque    (Nm)

    Foot forces — world frame  [FR, FL, RR, RL], xyz per foot:
      simforceFeetGlobal0.txt   FR_Fx
      simforceFeetGlobal1.txt   FR_Fy
      simforceFeetGlobal2.txt   FR_Fz
      simforceFeetGlobal3.txt   FL_Fx  ...  (and so on to index 11)

    Foot positions — world frame  [FR, FL, RR, RL], xyz per foot:
      footPosFeetGlobal0.txt    FR_x
      footPosFeetGlobal1.txt    FR_y
      footPosFeetGlobal2.txt    FR_z
      footPosFeetGlobal3.txt    FL_x  ...  (and so on to index 11)

    Contacts:
      contact_FR/FL/RR/RL.txt   binary (0/1)
      desPosTorso9.txt           front_stance (5 if FR or FL in contact, else 0)
      desPosTorso10.txt          rear_stance

    Misc:
      time.txt           simulation time (s)
      desired_vel_x.txt  commanded forward velocity (m/s)
      metadata.txt       human-readable notes
    """
    if data_count == 0:
        print("No data to save."); return False

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving {data_count} samples to {output_dir}...")

    def save1d(filename, arr):
        """Save a 1-D array: one float per line."""
        np.savetxt(os.path.join(output_dir, filename),
                   arr[:data_count], fmt='%.6f')

    # ── Torso ─────────────────────────────────────────────────────────────────
    save1d('PosTorso0.txt', data_arrays['torso_x'])
    save1d('PosTorso1.txt', data_arrays['torso_pitch'])
    save1d('PosTorso2.txt', data_arrays['torso_roll'])
    save1d('PosTorso3.txt', data_arrays['torso_yaw'])
    save1d('PosTorso4.txt', data_arrays['torso_y'])
    save1d('PosTorso5.txt', data_arrays['torso_z'])
    save1d('PosTorso6.txt', data_arrays['torso_vx'])
    save1d('PosTorso7.txt', data_arrays['torso_vy'])
    save1d('PosTorso8.txt', data_arrays['torso_vz'])

    # ── Joints (one scalar file per component) ────────────────────────────────
    for i in range(12):
        save1d(f'q{i}.txt',    data_arrays['joint_pos'][:, i])
        save1d(f'dq{i}.txt',   data_arrays['joint_vel'][:, i])
        save1d(f'tauM{i}.txt', data_arrays['joint_torque'][:, i])

    # ── Foot contact forces — world frame (one scalar per file) ───────────────
    # foot_forces shape: (N, 12)  layout: [FR_Fx, FR_Fy, FR_Fz, FL_Fx, ...]
    for i in range(12):
        save1d(f'simforceFeetGlobal{i}.txt', data_arrays['foot_forces'][:, i])

    # ── Foot positions — world frame (one scalar per file) ────────────────────
    # foot_pos shape: (N, 12)  layout: [FR_x, FR_y, FR_z, FL_x, FL_y, FL_z, ...]
    for i in range(12):
        save1d(f'footPosFeetGlobal{i}.txt', data_arrays['foot_pos'][:, i])

    # ── Contact states ────────────────────────────────────────────────────────
    for i, name in enumerate(['FR', 'FL', 'RR', 'RL']):
        save1d(f'contact_{name}.txt', data_arrays['contacts'][:, i])
    save1d('desPosTorso9.txt',  data_arrays['front_stance'])
    save1d('desPosTorso10.txt', data_arrays['rear_stance'])

    # ── Misc ─────────────────────────────────────────────────────────────────
    save1d('time.txt',           data_arrays['time'])
    save1d('desired_vel_x.txt',  data_arrays['desired_vel_x'])

    # ── Metadata ──────────────────────────────────────────────────────────────
    with open(os.path.join(output_dir, 'metadata.txt'), 'w') as f:
        f.write(f"# Gait: {_exit_state['gait_name']}\n")
        f.write(f"# Samples: {data_count}\n")
        f.write(f"# dt: 0.002s (500 Hz)\n")
        f.write(f"# Duration: {data_count * 0.002:.2f}s\n")
        f.write("#\n")
        f.write("# Each .txt file contains ONE FLOAT PER LINE.\n")
        f.write("# Foot index mapping (i = 0..11 in groups of 3):\n")
        f.write("#   i=0,1,2  -> FR  (x/Fx, y/Fy, z/Fz)\n")
        f.write("#   i=3,4,5  -> FL\n")
        f.write("#   i=6,7,8  -> RR\n")
        f.write("#   i=9,10,11-> RL\n")
        f.write("# simforceFeetGlobal<i>: contact force  component i (world frame, N)\n")
        f.write("# footPosFeetGlobal<i>:  contact position component i (world frame, m)\n")

    print(f"Saved to {output_dir}")
    return True


def create_video_from_frames(frames_dir, video_path, fps, frame_count):
    if frame_count == 0:
        return False
    print(f"\nCreating video from {frame_count} frames...")
    try:
        cmd = ['ffmpeg', '-y', '-framerate', str(fps),
               '-i', os.path.join(frames_dir, 'frame_%06d.png'),
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
               '-pix_fmt', 'yuv420p', video_path]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Video saved: {video_path}")
        shutil.rmtree(frames_dir)
        return True
    except Exception as e:
        print(f"✗ Video error: {e}")
        return False


def save_data_on_exit(reason="unknown"):
    global _exit_state, _frame_count
    if _exit_state['data_saved']:
        return
    _exit_state['data_saved'] = True
    print(f"\n{'='*60}\nSaving data (reason: {reason})...\n{'='*60}")
    elapsed = time.time() - _exit_state['start_time'] if _exit_state['start_time'] else 0
    print(f"Steps: {_exit_state['steps_count']}, Time: {elapsed:.2f}s, "
          f"Samples: {_exit_state['data_count']}")
    if _frame_count > 0 and _exit_state.get('frames_dir'):
        create_video_from_frames(_exit_state['frames_dir'],
                                 _exit_state['video_path'],
                                 FLAGS.render_fps, _frame_count)
    if _exit_state['data_count'] > 0 and _exit_state['data_arrays'] is not None:
        try:
            save_data_to_files(_exit_state['output_dir'],
                               _exit_state['data_arrays'],
                               _exit_state['data_count'])
        except Exception as e:
            print(f"ERROR saving: {e}")
            fallback = f"data_emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            save_data_to_files(fallback, _exit_state['data_arrays'],
                               _exit_state['data_count'])


def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}!")
    save_data_on_exit(reason=f"signal {signum}")
    sys.exit(0)

signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


from src.envs import env_wrappers
torch.set_printoptions(precision=2, sci_mode=False)

flags.DEFINE_string("logdir", None, "logdir.")
flags.DEFINE_bool("use_gpu", False, "whether to use GPU.")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_bool("use_real_robot", False, "whether to use real robot.")
flags.DEFINE_integer("num_envs", 1, "number of environments to evaluate in parallel.")
flags.DEFINE_bool("use_contact_sensor", True, "whether to use contact sensor.")
flags.DEFINE_integer("max_steps", 25000, "maximum number of simulation steps.")
flags.DEFINE_bool("record_video", False, "whether to record video of simulation.")
flags.DEFINE_integer("render_fps", 60, "FPS for video recording.")
flags.DEFINE_bool("render_landing_pos", True, "whether to render landing position boxes.")

FLAGS = flags.FLAGS


def get_latest_policy_path(logdir):
    files = [e for e in os.listdir(logdir) if os.path.isfile(os.path.join(logdir, e))]
    files.sort(key=lambda e: os.path.getmtime(os.path.join(logdir, e)), reverse=True)
    for e in files:
        if e.startswith("model"):
            return os.path.join(logdir, e)
    raise ValueError("No valid policy found.")


def main(argv):
    global _exit_state, _frame_count, _next_frame_time, _frame_interval
    global _frame_template, _gym, _viewer, _env
    del argv

    device = "cuda" if FLAGS.use_gpu else "cpu"

    if FLAGS.logdir.endswith("pt"):
        config_path = os.path.join(os.path.dirname(FLAGS.logdir), "config.yaml")
        policy_path = FLAGS.logdir
        root_path   = os.path.dirname(FLAGS.logdir)
    else:
        config_path = os.path.join(FLAGS.logdir, "config.yaml")
        policy_path = get_latest_policy_path(FLAGS.logdir)
        root_path   = FLAGS.logdir

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    gait_name = getattr(config.environment.gait, 'gait_name', 'unknown_gait')
    _exit_state['gait_name'] = gait_name

    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(root_path, f"detailed_{gait_name}_data_{timestamp}")
    _exit_state['output_dir'] = output_dir
    print(f"Gait: {gait_name}\nOutput: {output_dir}")

    with config.unlocked():
        velocity_up       = torch.linspace(2.5, 6.0, 50)
        velocity_schedule = velocity_up
        config.environment.jumping_distance_schedule = (
            velocity_schedule / config.environment.gait.stepping_frequency)
        config.environment.gait.desired_velocity = torch.tensor(
            [velocity_schedule[0].item(), 0, 0])
        config.environment.max_jumps = 100000

    show_gui_actual = FLAGS.show_gui or FLAGS.record_video
    env = config.env_class(
        num_envs=FLAGS.num_envs, device=device,
        config=config.environment, show_gui=show_gui_actual,
        use_real_robot=FLAGS.use_real_robot)
    env = env_wrappers.RangeNormalize(env)
    unwrapped_env = env._env
    _env = unwrapped_env
    print(f"Initial stepping_frequency: {unwrapped_env._gait_generator.stepping_frequency}")

    if FLAGS.use_real_robot:
        env.robot.state_estimator.use_external_contact_estimator = (
            not FLAGS.use_contact_sensor)

    if FLAGS.record_video:
        os.makedirs(output_dir, exist_ok=True)
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        video_path = os.path.join(
            output_dir,
            f"detailed_{gait_name}_data_{timestamp}_{FLAGS.render_fps}fps.mp4")
        _frame_count      = 0
        _next_frame_time  = 0.0
        _frame_interval   = 1.0 / FLAGS.render_fps
        _frame_template   = os.path.join(frames_dir, "frame_{:06d}.png")
        _gym    = unwrapped_env._gym
        _viewer = unwrapped_env._viewer
        _exit_state['frames_dir'] = frames_dir
        _exit_state['video_path'] = video_path
        unwrapped_env._frame_callback = capture_frame
        print(f"Frame capture: {FLAGS.render_fps} fps → {video_path}")

    runner = OnPolicyRunner(env, config.training, policy_path, device=device)
    runner.load(policy_path)
    policy = runner.get_inference_policy()
    runner.alg.actor_critic.train()

    state, _ = env.reset()

    N = FLAGS.max_steps + 1000
    data_arrays = {
        'time':          np.zeros(N,       dtype=np.float32),
        'torso_x':       np.zeros(N,       dtype=np.float32),
        'torso_y':       np.zeros(N,       dtype=np.float32),
        'torso_z':       np.zeros(N,       dtype=np.float32),
        'torso_vx':      np.zeros(N,       dtype=np.float32),
        'torso_vy':      np.zeros(N,       dtype=np.float32),
        'torso_vz':      np.zeros(N,       dtype=np.float32),
        'torso_roll':    np.zeros(N,       dtype=np.float32),
        'torso_pitch':   np.zeros(N,       dtype=np.float32),
        'torso_yaw':     np.zeros(N,       dtype=np.float32),
        'joint_pos':     np.zeros((N, 12), dtype=np.float32),
        'joint_vel':     np.zeros((N, 12), dtype=np.float32),
        'joint_torque':  np.zeros((N, 12), dtype=np.float32),
        # Foot forces: (N, 12) = [FR_Fx, FR_Fy, FR_Fz, FL_Fx, ..., RL_Fz]
        # Source: robot.foot_contact_forces  shape (num_envs, 4, 3)
        'foot_forces':   np.zeros((N, 12), dtype=np.float32),
        # Foot positions in world frame: (N, 12) = [FR_x, FR_y, FR_z, FL_x, ...]
        # Source: robot.foot_positions_in_world_frame  shape (num_envs, 4, 3)
        'foot_pos':      np.zeros((N, 12), dtype=np.float32),
        'contacts':      np.zeros((N,  4), dtype=np.float32),
        'front_stance':  np.zeros(N,       dtype=np.float32),
        'rear_stance':   np.zeros(N,       dtype=np.float32),
        'desired_vel_x': np.zeros(N,       dtype=np.float32),
    }
    _exit_state['data_arrays'] = data_arrays
    _exit_state['start_time']  = time.time()

    print(f"Starting simulation (max {FLAGS.max_steps} steps)...")

    steps_count    = 0
    data_count     = 0
    velocity_index = 0
    pbar = tqdm(total=FLAGS.max_steps, desc="Evaluating", unit="step")

    try:
        with torch.inference_mode():
            while steps_count < FLAGS.max_steps:
                steps_count += 1
                action = policy(state)
                state, _, reward, done, info = env.step(action)

                # ── Extract robot state ────────────────────────────────────────
                t         = env.robot.time_since_reset.item()
                contacts  = env.robot.foot_contacts[0].cpu().numpy().astype(np.float32)
                base_pos  = env.robot.base_position[0].cpu().numpy()
                base_vel  = env.robot.base_velocity_world_frame[0].cpu().numpy()
                base_rpy  = env.robot.base_orientation_rpy[0].cpu().numpy()
                joint_pos = env.robot.motor_positions[0].cpu().numpy()
                joint_vel = env.robot.motor_velocities[0].cpu().numpy()
                joint_tau = env.robot.motor_torques[0].cpu().numpy()

                # ── Foot contact forces — world frame ──────────────────────────
                # robot.foot_contact_forces: (num_envs, 4, 3)  [FR, FL, RR, RL]
                foot_forces = (env.robot.foot_contact_forces[0]
                               .cpu().numpy()          # shape (4, 3)
                               .flatten())             # → 12 scalars

                # ── Foot positions — world frame ───────────────────────────────
                # robot.foot_positions_in_world_frame: (num_envs, 4, 3)
                foot_pos = (env.robot.foot_positions_in_world_frame[0]
                            .cpu().numpy()             # shape (4, 3)
                            .flatten())                # → 12 scalars

                # ── Store ──────────────────────────────────────────────────────
                i = data_count
                data_arrays['time'][i]          = t
                data_arrays['torso_x'][i]       = base_pos[0]
                data_arrays['torso_y'][i]       = base_pos[1]
                data_arrays['torso_z'][i]       = base_pos[2]
                data_arrays['torso_vx'][i]      = base_vel[0]
                data_arrays['torso_vy'][i]      = base_vel[1]
                data_arrays['torso_vz'][i]      = base_vel[2]
                data_arrays['torso_roll'][i]    = base_rpy[0]
                data_arrays['torso_pitch'][i]   = base_rpy[1]
                data_arrays['torso_yaw'][i]     = base_rpy[2]
                data_arrays['joint_pos'][i]     = joint_pos
                data_arrays['joint_vel'][i]     = joint_vel
                data_arrays['joint_torque'][i]  = joint_tau
                data_arrays['foot_forces'][i]   = foot_forces
                data_arrays['foot_pos'][i]      = foot_pos
                data_arrays['contacts'][i]      = contacts
                data_arrays['front_stance'][i]  = 5.0 if (contacts[0] or contacts[1]) else 0.0
                data_arrays['rear_stance'][i]   = 5.0 if (contacts[2] or contacts[3]) else 0.0
                data_arrays['desired_vel_x'][i] = velocity_schedule[velocity_index].item()

                # ── Progress ───────────────────────────────────────────────────
                curr_vel  = np.linalg.norm(base_vel[:2])
                real_time = time.time() - _exit_state['start_time']
                freq      = unwrapped_env._gait_generator.stepping_frequency[0].cpu().numpy()
                pbar.set_postfix({
                    'SimTime':  f'{t:.2f}s',
                    'RealTime': f'{real_time:.2f}s',
                    'Freq':     f'{freq:.2f}Hz',
                    'CurrVel':  f'{curr_vel:.3f}m/s',
                    'DesVel':   f'{velocity_schedule[velocity_index].item():.3f}m/s',
                    'Frames':   _frame_count if FLAGS.record_video else 0,
                })
                pbar.update(1)

                data_count += 1
                _exit_state['data_count']  = data_count
                _exit_state['steps_count'] = steps_count

                if steps_count % 50 == 0 and velocity_index < len(velocity_schedule) - 1:
                    velocity_index += 1
                    env._env._desired_velocity[:, 0] = velocity_schedule[velocity_index]
                    env._env._desired_velocity[:, 1:] = 0

    except Exception as e:
        pbar.close()
        print(f"\nException: {e}")
        import traceback; traceback.print_exc()
        save_data_on_exit(reason="exception")
        raise

    pbar.close()
    save_data_on_exit(reason="complete")


if __name__ == "__main__":
    app.run(main)
