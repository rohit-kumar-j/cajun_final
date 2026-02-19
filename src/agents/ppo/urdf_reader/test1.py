"""
Go2 Robot Viewer - test1.py

Backends:
  --backend pyglet  (default) - OpenGL 3.3, STL meshes, ImGui panel
  --backend dpg               - DearPyGui software projection, stick figure

Usage:
    python test1.py --data g2_rotary_data --urdf go2/urdf/go2.urdf
    python test1.py --data g2_rotary_data --urdf go2/urdf/go2.urdf --backend dpg

Controls:
    Space        play / pause
    Left / Right step one frame
    LMB drag     orbit  (viewport only)
    RMB drag     pan    (viewport only)
    Scroll       zoom   (viewport only)
    ESC          quit
"""

import argparse, os, sys, math, time
import numpy as np
from pathlib import Path
from enum import Enum, auto


# =============================================================================
#  MOUSE REGION ENUM
# =============================================================================

class Region(Enum):
    VIEWPORT = auto()
    PANEL    = auto()
    UNKNOWN  = auto()


# =============================================================================
#  DATA LOADER
# =============================================================================

def load_column(path):
    return np.loadtxt(path, dtype=np.float64)

def load_data(data_dir):
    d = Path(data_dir)
    def col(name):
        p = d / name
        return load_column(p) if p.exists() else None

    time_arr = col('time.txt')
    if time_arr is None:
        raise FileNotFoundError(f"time.txt not found in {data_dir}")
    n = len(time_arr)

    data = {
        'time':          time_arr,
        'torso_x':       col('PosTorso0.txt'),
        'torso_pitch':   col('PosTorso1.txt'),
        'torso_roll':    col('PosTorso2.txt'),
        'torso_yaw':     col('PosTorso3.txt'),
        'torso_y':       col('PosTorso4.txt'),
        'torso_z':       col('PosTorso5.txt'),
        'desired_vel_x': col('desired_vel_x.txt'),
        'contact_FL':    col('contact_FL.txt'),
        'contact_FR':    col('contact_FR.txt'),
        'contact_RL':    col('contact_RL.txt'),
        'contact_RR':    col('contact_RR.txt'),
    }
    data['q']           = np.zeros((n, 12))
    data['dq']          = np.zeros((n, 12))
    data['tau']         = np.zeros((n, 12))
    data['foot_forces'] = np.zeros((n, 12))
    data['foot_pos']    = np.zeros((n, 12))   # world-frame foot contact positions
    for i in range(12):
        for prefix, key in [('q','q'),('dq','dq'),('tauM','tau'),
                             ('simforceFeetGlobal','foot_forces'),
                             ('footPosFeetGlobal','foot_pos')]:
            v = col(f'{prefix}{i}.txt')
            if v is not None:
                data[key][:, i] = v

    dt = time_arr[1] - time_arr[0]
    print(f"Loaded {n} frames  dt={dt:.4f}s  "
          f"duration={time_arr[-1]-time_arr[0]:.2f}s")
    return data, n


# =============================================================================
#  URDF PARSER  — reads joint limits and offsets directly from URDF XML
# =============================================================================

def parse_urdf(urdf_path):
    """
    Parse URDF and return:
      joint_order   : list of actuated joint names (in order)
      joint_limits  : dict  name -> (lower, upper)
      joint_axes    : dict  name -> np.array([x,y,z])
      joint_origins : dict  name -> (xyz np.array, rpy np.array)
      link_parents  : dict  link_name -> parent_link_name
    Raises on failure — no silent defaults.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joint_order   = []
    joint_limits  = {}
    joint_axes    = {}
    joint_origins = {}
    link_parents  = {}

    def parse_vec3(s, default='0 0 0'):
        parts = (s or default).split()
        return np.array([float(x) for x in parts], dtype=np.float64)

    for joint in root.findall('joint'):
        jtype = joint.get('type', 'fixed')
        name  = joint.get('name')
        parent = joint.find('parent')
        child  = joint.find('child')
        if parent is not None and child is not None:
            link_parents[child.get('link')] = parent.get('link')

        if jtype not in ('revolute', 'continuous', 'prismatic'):
            continue

        origin = joint.find('origin')
        xyz = parse_vec3(origin.get('xyz') if origin is not None else None)
        rpy = parse_vec3(origin.get('rpy') if origin is not None else None)
        joint_origins[name] = (xyz, rpy)

        axis_el = joint.find('axis')
        joint_axes[name] = parse_vec3(
            axis_el.get('xyz') if axis_el is not None else None,
            default='1 0 0')

        limit_el = joint.find('limit')
        if limit_el is not None:
            lo = float(limit_el.get('lower', -1e9))
            hi = float(limit_el.get('upper',  1e9))
            joint_limits[name] = (lo, hi)
        else:
            joint_limits[name] = (-1e9, 1e9)

        joint_order.append(name)

    if not joint_order:
        raise ValueError(f"No actuated joints found in {urdf_path}")

    print(f"URDF: parsed {len(joint_order)} joints from {urdf_path}")
    return joint_order, joint_limits, joint_axes, joint_origins, link_parents


# =============================================================================
#  KINEMATICS  — built from parsed URDF data
# =============================================================================

def rot_axis_angle(axis, angle):
    c, s = math.cos(float(angle)), math.sin(float(angle))
    t = 1.0 - c
    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    return np.array([
        [t*x*x+c,   t*x*y-s*z, t*x*z+s*y],
        [t*x*y+s*z, t*y*y+c,   t*y*z-s*x],
        [t*x*z-s*y, t*y*z+s*x, t*z*z+c  ],
    ], dtype=np.float64)

def euler_to_rot(roll, pitch, yaw):
    return (rot_axis_angle([0,0,1], yaw)
          @ rot_axis_angle([0,1,0], pitch)
          @ rot_axis_angle([1,0,0], roll))

def rpy_to_rot(rpy):
    return euler_to_rot(rpy[0], rpy[1], rpy[2])

def is_near_limit(lo, hi, val, tol=0.05):
    span = abs(hi - lo)
    if span < 1e-9:
        return False
    return (val - lo) < tol * span or (hi - val) < tol * span


# Detect leg structure from joint names
def detect_legs(joint_order):
    """
    Returns dict: leg_label -> list_of_joint_indices
    Groups joints by leg prefix heuristic.
    """
    from collections import OrderedDict
    legs = OrderedDict()
    for i, name in enumerate(joint_order):
        # e.g. '1_FR_hip_joint' -> prefix '1_FR'
        parts = name.split('_')
        if len(parts) >= 2:
            prefix = '_'.join(parts[:2])  # '1_FR'
            label  = parts[1]             # 'FR'
        else:
            prefix = parts[0]
            label  = parts[0]
        if label not in legs:
            legs[label] = []
        legs[label].append(i)
    return legs  # e.g. {'FR':[0,1,2], 'FL':[3,4,5], ...}

LEG_COLORS_F = {
    'FR': (0.30, 0.90, 0.30, 1.0),
    'FL': (0.30, 0.55, 1.00, 1.0),
    'RR': (1.00, 0.63, 0.16, 1.0),
    'RL': (0.90, 0.24, 0.90, 1.0),
}
LEG_COLORS_I = {
    'FR': ( 80, 220,  80, 255),
    'FL': ( 80, 140, 255, 255),
    'RR': (255, 160,  40, 255),
    'RL': (230,  60, 230, 255),
}

def get_leg_color_f(label):
    return LEG_COLORS_F.get(label, (0.7, 0.7, 0.7, 1.0))

def get_leg_color_i(label):
    return LEG_COLORS_I.get(label, (180, 180, 180, 255))


# =============================================================================
#  FK using yourdfpy
# =============================================================================

def build_fk_fn(urdf_path, joint_order):
    """
    Returns a function:
        fk(q_vec, torso_pos, torso_rpy) -> dict link_name -> 4x4 world transform
    Uses yourdfpy for correct visual transforms.
    """
    try:
        from yourdfpy import URDF as _URDF
        robot = _URDF.load(urdf_path)
        print(f"FK: yourdfpy loaded {urdf_path}")
    except Exception as e:
        raise ImportError(f"yourdfpy required for FK: {e}")

    ref = 'trunk' if 'trunk' in robot.link_map else robot.base_link

    def fk(q_vec, torso_pos, torso_rpy):
        # update_cfg MUST happen before any scene graph access
        cfg = {jname: float(q_vec[i])
               for i, jname in enumerate(joint_order)}
        robot.update_cfg(cfg)

        R_world = euler_to_rot(*torso_rpy).astype(np.float32)
        T_world = np.eye(4, dtype=np.float32)
        T_world[:3,:3] = R_world
        T_world[:3, 3] = torso_pos

        transforms = {}

        # Primary: use yourdfpy scene graph which has visual origins baked in
        # After update_cfg() the graph holds the current pose including visual frames
        try:
            scene = robot.scene
            for node_name in scene.graph.nodes_geometry:
                T_node, _ = scene.graph.get(node_name)
                if T_node is not None:
                    # node names are like "trunk/visuals/0" or "1_FR_hip/visuals/0"
                    link_name = node_name.split('/')[0]
                    # Keep first visual per link (index 0), skip duplicates
                    if link_name not in transforms:
                        transforms[link_name] = T_world @ T_node.astype(np.float32)
        except Exception as _e:
            pass  # fall through to link-frame fallback

        # Fallback: link origin transforms (no visual offset, used for joint positions)
        for link in robot.link_map:
            if link not in transforms:
                try:
                    T = robot.get_transform(link, ref).astype(np.float32)
                    transforms[link] = T_world @ T
                except Exception:
                    pass
        return transforms

    LINK_TO_MESH_NAMES = {'trunk','1_FR_hip','2_FL_hip','3_RR_hip','4_RL_hip',
                          '1_FR_thigh','2_FL_thigh','3_RR_thigh','4_RL_thigh',
                          '1_FR_calf','2_FL_calf','3_RR_calf','4_RL_calf'}
    return fk, robot


# =============================================================================
#  SHARED APP STATE
# =============================================================================

class AppState:
    def __init__(self):
        self.data         = None
        self.n_frames     = 0
        self.frame        = 0
        self.playing      = False
        self.play_speed   = 0.5
        self._last_t      = 0.0
        self._frac_acc    = 0.0

        # loop pins
        self.loop_start   = 0
        self.loop_end     = 0   # 0 = use n_frames-1

        # camera
        self.cam_yaw      = 45.0
        self.cam_pitch    = 25.0
        self.cam_dist     = 3.5
        self.cam_target   = np.array([0.0, 0.0, 0.3])
        self.cam_follow_mode = 'main'

        # overlays
        self.show_forces      = True
        self.show_contacts    = True
        self.show_limits      = True
        self.show_grid        = True
        self.show_trajectory  = False
        self.show_joint_frames = False   # F6 toggles
        self.frame_cycle       = 0       # 0=none,1=joint,2=body (F6 cycles)
        self.traj_length      = 200
        self.force_scale      = 0.003
        self.fog_start        = 3.0
        self.fog_end          = 18.0

        # mouse
        self.mouse_region = Region.UNKNOWN

    def _hi(self):
        return (self.loop_end if self.loop_end > 0
                else self.n_frames - 1)

    def advance(self):
        if G.play_mode == 'graph':
            return
        if not self.playing or self.data is None:
            return
        now = time.time()
        dt_real = now - self._last_t
        self._last_t = now
        dt_sim = float(self.data['time'][1] - self.data['time'][0])
        self._frac_acc += dt_real * self.play_speed / dt_sim
        steps = int(self._frac_acc)
        self._frac_acc -= steps
        lo   = self.loop_start
        hi   = self._hi()
        span = max(hi - lo + 1, 2)
        self.frame = lo + (self.frame - lo + steps) % span
        self._sync_cam()

    def update_cam(self):
        if self.data is None: return
        f = G.frame if self.cam_follow_mode == 'ghost' else self.frame
        self.cam_target = np.array([
            float(self.data['torso_x'][f]),
            float(self.data['torso_y'][f]),
            float(self.data['torso_z'][f]),
        ])

    def _sync_cam(self, f=None):
        self.update_cam()

    def step(self, delta):
        lo = self.loop_start
        hi = self._hi()
        self.frame = max(lo, min(hi, self.frame + delta))
        self.update_cam()
        G.sync_to_main()

    def toggle_play(self):
        self.playing   = not self.playing
        self._last_t   = time.time()
        self._frac_acc = 0.0

    def cam_eye(self):
        yr = math.radians(self.cam_yaw)
        pr = math.radians(max(5.0, min(85.0, self.cam_pitch)))
        return self.cam_target + self.cam_dist * np.array([
            math.cos(pr)*math.cos(yr),
            math.cos(pr)*math.sin(yr),
            math.sin(pr),
        ])

    def frame_state(self, f=None):
        if f is None:
            f = self.frame
        d = self.data
        q   = d['q'][f]
        pos = np.array([float(d['torso_x'][f]),
                        float(d['torso_y'][f]),
                        float(d['torso_z'][f])])
        rpy = (float(d['torso_roll'][f]),
               float(d['torso_pitch'][f]),
               float(d['torso_yaw'][f]))
        return q, pos, rpy


S = AppState()

class GraphScrubState:
    """
    Shared scrub position for all live graphs.
    Two play modes:
      'main'  — G follows S.frame; main loop plays normally
      'graph' — G plays its own ±50 loop; S.frame frozen at backup_frame
    """
    def __init__(self):
        self.frame        = 0
        self.frozen       = False
        self.play_mode    = 'main'
        self.backup_frame = 0
        self._playing     = False
        self._last_t      = 0.0
        self._frac_acc    = 0.0

    def sync_to_main(self):
        if self.play_mode == 'main':
            self.frame  = S.frame
            self.frozen = False
        else:
            self.backup_frame = S.frame

    def scrub(self, fi, n_frames):
        anchor = self.backup_frame if self.play_mode == 'graph' else S.frame
        lo = max(0, anchor - 50)
        hi = min(n_frames - 1, anchor + 50)
        self.frame  = max(lo, min(hi, fi))
        self.frozen = True
        S._sync_cam(self.frame)

    def release(self):
        self.frozen = False

    def set_mode(self, mode):
        if mode == self.play_mode:
            return
        if mode == 'graph':
            self.backup_frame = S.frame
            self.frame        = S.frame
            self._playing     = False
            self.play_mode    = 'graph'
        else:
            self.play_mode    = 'main'
            self._playing     = False
            S.frame           = self.backup_frame
            S.update_cam()
            self.frame        = S.frame
            self.frozen       = False

    def toggle_graph_play(self, n_frames):
        if self.play_mode != 'graph':
            return
        self._playing  = not self._playing
        self._last_t   = time.time()
        self._frac_acc = 0.0

    def advance_graph(self, n_frames, play_speed, dt_sim):
        if self.play_mode != 'graph' or not self._playing:
            return
        now = time.time()
        dt_real = now - self._last_t
        self._last_t = now
        self._frac_acc += dt_real * play_speed / dt_sim
        steps = int(self._frac_acc)
        self._frac_acc -= steps
        lo = max(0, self.backup_frame - 50)
        hi = min(n_frames - 1, self.backup_frame + 50)
        span = max(hi - lo + 1, 2)
        self.frame = lo + (self.frame - lo + steps) % span
        S.update_cam()

G = GraphScrubState()




# =============================================================================
#  BACKEND A: PYGLET + OPENGL 3.3 + IMGUI
# =============================================================================

def run_pyglet(urdf_path, joint_order, joint_limits, joint_axes, joint_origins, fk_fn, urdf_robot):
    try:
        import pyglet
        from pyglet.gl import (
            glEnable, glDisable, glClear, glClearColor, glViewport,
            glDepthFunc, glBlendFunc,
            GL_DEPTH_TEST, GL_BLEND, GL_LEQUAL,
            GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
            GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
            GL_TRIANGLES, GL_LINES,
            glGenVertexArrays, glBindVertexArray,
            glGenBuffers, glBindBuffer, glBufferData,
            glEnableVertexAttribArray, glVertexAttribPointer,
            glDrawArrays, glDrawElements,
            glUseProgram, glGetUniformLocation,
            glUniformMatrix4fv, glUniform3f, glUniform4f,
            glUniform1f, glUniform1i,
            GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER,
            GL_STATIC_DRAW, GL_DYNAMIC_DRAW,
            GL_FLOAT, GL_UNSIGNED_INT, GL_FALSE, GL_TRUE,
            glScissor
        )
        import ctypes
    except ImportError as e:
        print(f"pyglet error: {e}")
        sys.exit(1)

    try:
        import imgui
    except ImportError:
        try:
            from imgui_bundle import imgui
        except ImportError:
            print("imgui not installed: pip install 'imgui[pyglet]'")
            sys.exit(1)

    import trimesh

    # ── Shaders ───────────────────────────────────────────────────────────────

    VERT_LIT = b"""
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 uMVP;
uniform mat4 uModel;
uniform mat3 uNormalMat;
out vec3 vNormal;
out vec3 vFragPos;
void main(){
    vFragPos    = (uModel * vec4(aPos,1)).xyz;
    vNormal     = normalize(uNormalMat * aNormal);
    gl_Position = uMVP * vec4(aPos,1);
}
"""
    FRAG_LIT = b"""
#version 330 core
in vec3 vNormal; in vec3 vFragPos;
uniform vec3 uColor; uniform vec3 uLightPos; uniform vec3 uCamPos;
uniform float uAlpha; uniform float uFogStart; uniform float uFogEnd;
out vec4 FragColor;
void main(){
    vec3 n   = normalize(vNormal);
    vec3 l   = normalize(uLightPos - vFragPos);
    vec3 v   = normalize(uCamPos   - vFragPos);
    vec3 h   = normalize(l+v);
    float amb = 0.32;
    float dif = max(dot(n,l),0.0)*0.68;
    float spc = pow(max(dot(n,h),0.0),48.0)*0.30;
    vec3 col  = uColor*(amb+dif)+vec3(spc);
    // fog
    float d   = length(vFragPos - uCamPos);
    float fog = clamp((d-uFogStart)/(uFogEnd-uFogStart),0.0,1.0);
    vec3 fogCol = vec3(0.52,0.60,0.68);
    col = mix(col, fogCol, fog);
    FragColor = vec4(col, uAlpha);
}
"""
    VERT_FLAT = b"""
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP; uniform vec4 uColor;
out vec4 vColor;
void main(){ vColor=uColor; gl_Position=uMVP*vec4(aPos,1); }
"""
    FRAG_FLAT = b"""
#version 330 core
in vec4 vColor; out vec4 FragColor;
void main(){ FragColor=vColor; }
"""

    # ── Shader compiler ───────────────────────────────────────────────────────

    from pyglet.gl import (
        glCreateShader, glShaderSource, glCompileShader, glGetShaderiv,
        glGetShaderInfoLog, glCreateProgram, glAttachShader, glLinkProgram,
        glGetProgramiv, glGetProgramInfoLog, glDeleteShader,
        GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
        GL_COMPILE_STATUS, GL_LINK_STATUS, GL_INFO_LOG_LENGTH,
    )

    def _compile(src, stype):
        sid = glCreateShader(stype)
        buf = ctypes.create_string_buffer(src)
        ptr = ctypes.cast(ctypes.pointer(ctypes.pointer(buf)),
                          ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
        glShaderSource(sid, 1, ptr, None)
        glCompileShader(sid)
        ok = ctypes.c_int(0); glGetShaderiv(sid, GL_COMPILE_STATUS, ctypes.byref(ok))
        if not ok.value:
            n = ctypes.c_int(0); glGetShaderiv(sid, GL_INFO_LOG_LENGTH, ctypes.byref(n))
            log = (ctypes.c_char*n.value)(); glGetShaderInfoLog(sid,n.value,None,log)
            raise RuntimeError(f"Shader error: {log.value.decode()}")
        return sid

    def _link(vsrc, fsrc):
        vs = _compile(vsrc, GL_VERTEX_SHADER)
        fs = _compile(fsrc, GL_FRAGMENT_SHADER)
        p  = glCreateProgram()
        glAttachShader(p,vs); glAttachShader(p,fs); glLinkProgram(p)
        ok = ctypes.c_int(0); glGetProgramiv(p,GL_LINK_STATUS,ctypes.byref(ok))
        if not ok.value:
            n = ctypes.c_int(0); glGetProgramiv(p,GL_INFO_LOG_LENGTH,ctypes.byref(n))
            log=(ctypes.c_char*n.value)(); glGetProgramInfoLog(p,n.value,None,log)
            raise RuntimeError(f"Link error: {log.value.decode()}")
        glDeleteShader(vs); glDeleteShader(fs)
        return p

    # ── Matrix helpers ────────────────────────────────────────────────────────

    def _persp(fov, asp, near, far):
        f = 1.0/math.tan(math.radians(fov)/2.0)
        nf = 1.0/(near-far)
        return np.array([
            [f/asp,0,0,0],[0,f,0,0],
            [0,0,(far+near)*nf,2*far*near*nf],
            [0,0,-1,0]], dtype=np.float32)

    def _lookat(eye, tgt, up):
        eye=np.array(eye,dtype=np.float64)
        f=np.array(tgt,dtype=np.float64)-eye; f/=np.linalg.norm(f)
        u2=np.array(up,dtype=np.float64)
        if abs(np.dot(f,u2))>0.99: u2=np.array([0.,1.,0.])
        r=np.cross(f,u2); r/=np.linalg.norm(r); u2=np.cross(r,f)
        M=np.eye(4,dtype=np.float32)
        M[0,:3]=r;   M[0,3]=float(-np.dot(r, eye))
        M[1,:3]=u2;  M[1,3]=float(-np.dot(u2,eye))
        M[2,:3]=-f;  M[2,3]=float( np.dot(f, eye))
        return M

    def _u4(prog, name, M):
        loc=glGetUniformLocation(prog, name.encode())
        flat=np.ascontiguousarray(M,dtype=np.float32).flatten()
        glUniformMatrix4fv(loc,1,GL_TRUE,(ctypes.c_float*16)(*flat))

    def _u3f(prog, name, M):
        from pyglet.gl import glUniformMatrix3fv
        loc=glGetUniformLocation(prog, name.encode())
        flat=np.ascontiguousarray(M,dtype=np.float32).flatten()
        glUniformMatrix3fv(loc,1,GL_TRUE,(ctypes.c_float*9)(*flat))

    def _uf3(prog, name, x,y,z):
        glUniform3f(glGetUniformLocation(prog,name.encode()),x,y,z)

    def _uf4(prog, name, x,y,z,w):
        glUniform4f(glGetUniformLocation(prog,name.encode()),x,y,z,w)

    def _uf1(prog, name, v):
        glUniform1f(glGetUniformLocation(prog,name.encode()),v)

    # ── VAO builders ──────────────────────────────────────────────────────────

    def _vao_mesh(verts, norms, faces):
        data = np.hstack([verts,norms]).astype(np.float32).flatten()
        idx  = faces.flatten().astype(np.uint32)
        vao=ctypes.c_uint(0); glGenVertexArrays(1,ctypes.byref(vao))
        vbo=ctypes.c_uint(0); glGenBuffers(1,ctypes.byref(vbo))
        ebo=ctypes.c_uint(0); glGenBuffers(1,ctypes.byref(ebo))
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER,vbo)
        glBufferData(GL_ARRAY_BUFFER,data.nbytes,
                     data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),GL_STATIC_DRAW)
        st=6*ctypes.sizeof(ctypes.c_float)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,st,ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,st,ctypes.c_void_p(3*4))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,idx.nbytes,
                     idx.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),GL_STATIC_DRAW)
        glBindVertexArray(ctypes.c_uint(0))
        return vao, len(idx)

    def _vao_lines(max_pts=4096):
        buf=np.zeros((max_pts,3),dtype=np.float32)
        vao=ctypes.c_uint(0); glGenVertexArrays(1,ctypes.byref(vao))
        vbo=ctypes.c_uint(0); glGenBuffers(1,ctypes.byref(vbo))
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER,vbo)
        glBufferData(GL_ARRAY_BUFFER,buf.nbytes,
                     buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,12,ctypes.c_void_p(0))
        glBindVertexArray(ctypes.c_uint(0))
        return vao, vbo

    def _upload_lines(vbo, pts):
        arr=np.array(pts,dtype=np.float32).flatten()
        glBindBuffer(GL_ARRAY_BUFFER,vbo)
        glBufferData(GL_ARRAY_BUFFER,arr.nbytes,
                     arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),GL_DYNAMIC_DRAW)

    # ── Primitive mesh generators ─────────────────────────────────────────────

    def _sphere_mesh(r=1.0, st=10, sl=10):
        vv,nn,ff=[],[],[]
        for i in range(st+1):
            lat=math.pi*(-0.5+i/st)
            for j in range(sl+1):
                lon=2*math.pi*j/sl
                x=math.cos(lat)*math.cos(lon); y=math.cos(lat)*math.sin(lon); z=math.sin(lat)
                vv.append([x*r,y*r,z*r]); nn.append([x,y,z])
        for i in range(st):
            for j in range(sl):
                a=i*(sl+1)+j
                ff+=[[a,a+1,a+sl+1],[a+1,a+sl+2,a+sl+1]]
        return np.array(vv,np.float32),np.array(nn,np.float32),np.array(ff,np.uint32)

    def _ground_mesh_checker(size=26.0, step=0.5):
        """Returns (dark_vao_data, light_vao_data) separately for two-tone checker."""
        def build(parity):
            vv,nn,ff=[],[],[]
            rng=range(int(-size/step),int(size/step)+1)
            for xi in rng:
                for yi in rng:
                    if (xi+yi)%2 != parity: continue
                    x0,y0=xi*step,yi*step
                    b=len(vv)
                    for x,y in[(x0,y0),(x0+step,y0),(x0+step,y0+step),(x0,y0+step)]:
                        vv.append([x,y,0.0]); nn.append([0,0,1])
                    ff+=[[b,b+1,b+2],[b,b+2,b+3]]
            return (np.array(vv,np.float32),np.array(nn,np.float32),np.array(ff,np.uint32))
        return build(0), build(1)

    # keep old _ground_mesh for compatibility
    def _ground_mesh(size=22.0, step=0.5):
        vv,nn,ff=[],[],[]
        rng=range(int(-size/step),int(size/step)+1)
        for xi in rng:
            for yi in rng:
                x0,y0=xi*step,yi*step
                b=len(vv)
                for x,y in[(x0,y0),(x0+step,y0),(x0+step,y0+step),(x0,y0+step)]:
                    vv.append([x,y,0.0]); nn.append([0,0,1])
                ff+=[[b,b+1,b+2],[b,b+2,b+3]]
        return np.array(vv,np.float32),np.array(nn,np.float32),np.array(ff,np.uint32)

    # ── Window ────────────────────────────────────────────────────────────────

    W, H    = 1420, 820
    PANEL_W = 400

    cfg = pyglet.gl.Config(double_buffer=True, depth_size=24,
                           major_version=3, minor_version=3,
                           forward_compatible=True,
                           sample_buffers=1, samples=4)
    win = pyglet.window.Window(width=W, height=H,
                               caption='Go2 Viewer [pyglet]',
                               resizable=True, config=cfg)

    # ── Init GL (after window) ────────────────────────────────────────────────

    prog_lit  = _link(VERT_LIT,  FRAG_LIT)
    prog_flat = _link(VERT_FLAT, FRAG_FLAT)

    URDF_DIR = Path(urdf_path).parent
    MESH_DIR = URDF_DIR.parent / 'meshes'
    MESH_KEYS = ['base','hip','thigh','thigh_mirror','calf','calf_mirror']
    MESH_VAOS = {}
    for key in MESH_KEYS:
        p = MESH_DIR / f'{key}.stl'
        if p.exists():
            try:
                m = trimesh.load(str(p), force='mesh')
                MESH_VAOS[key] = _vao_mesh(
                    np.array(m.vertices,      dtype=np.float32),
                    np.array(m.vertex_normals, dtype=np.float32),
                    np.array(m.faces,          dtype=np.uint32))
                print(f"  mesh {key}: {len(m.faces)} tris")
            except Exception as ex:
                print(f"  mesh {key} failed: {ex}")

    LINK_TO_MESH = {
        'trunk':       'base',
        '1_FR_hip':    'hip',        '2_FL_hip':   'hip',
        '3_RR_hip':    'hip',        '4_RL_hip':   'hip',
        '1_FR_thigh':  'thigh_mirror','2_FL_thigh': 'thigh',
        '3_RR_thigh':  'thigh_mirror','4_RL_thigh': 'thigh',
        '1_FR_calf':   'calf_mirror', '2_FL_calf':  'calf',
        '3_RR_calf':   'calf_mirror', '4_RL_calf':  'calf',
    }

    VAO_SPHERE = _vao_mesh(*_sphere_mesh(1.0))
    (gv0,gn0,gf0),(gv1,gn1,gf1) = _ground_mesh_checker(26.0, 0.5)
    VAO_GROUND_DARK  = _vao_mesh(gv0,gn0,gf0)
    VAO_GROUND_LIGHT = _vao_mesh(gv1,gn1,gf1)
    VAO_LINES, VBO_LINES = _vao_lines()

    legs = detect_legs(joint_order)

    # ── ImGui setup ───────────────────────────────────────────────────────────

    imgui.create_context()
    _io = imgui.get_io()
    try:   _io.ini_file_name = b''
    except Exception: pass
    _io.display_size = (W, H)

    # Build imgui renderer manually (works with any imgui version)
    try:
        from imgui.integrations.opengl import ProgrammablePipelineRenderer as _R
        _imgui_renderer = _R()
    except Exception:
        try:
            from imgui_bundle.integrations.opengl import ProgrammablePipelineRenderer as _R
            _imgui_renderer = _R()
        except Exception as e:
            print(f"ImGui renderer error: {e}")
            sys.exit(1)

    # Mouse state tracked manually (pyglet 2.x broke create_renderer hooks)
    _mb = [False, False, False]   # left, middle, right
    _mx, _my = W//2, H//2

    # ── Render helpers ────────────────────────────────────────────────────────

    def _vp_w():
        return max(200, int(win.width * _layout['vp_frac']))

    def _matrices():
        vw   = _vp_w()
        bot  = max(80, int(win.height * _layout['bot_frac']))
        vh   = max(1, win.height - bot)   # actual viewport height (minus bottom bar)
        eye  = S.cam_eye()
        proj = _persp(60.0, vw/vh, 0.01, 500.0)
        view = _lookat(eye, S.cam_target, [0,0,1])
        return proj, view, eye

    def _draw_mesh(vao, n_idx, prog, model, color3, proj, view, eye, alpha=1.0):
        glUseProgram(prog)
        mvp = proj @ view @ model
        _u4(prog,'uMVP',   mvp)
        _u4(prog,'uModel', model)
        nm = np.linalg.inv(model[:3,:3]).T.astype(np.float32)
        _u3f(prog,'uNormalMat', nm)
        _uf3(prog,'uColor', *color3)
        _uf3(prog,'uLightPos', *(eye+np.array([0,0,3.0])))
        _uf3(prog,'uCamPos', *eye)
        _uf1(prog,'uAlpha', float(alpha))
        _uf1(prog,'uFogStart', S.fog_start)
        _uf1(prog,'uFogEnd',   S.fog_end)
        glBindVertexArray(vao)
        glDrawElements(GL_TRIANGLES,n_idx,GL_UNSIGNED_INT,ctypes.c_void_p(0))
        glBindVertexArray(ctypes.c_uint(0))

    def _draw_sphere(pos, r, col3, prog, proj, view, eye):
        s = np.diag([r,r,r,1.0]).astype(np.float32)
        T = np.eye(4,dtype=np.float32); T[:3,3]=pos
        _draw_mesh(VAO_SPHERE[0],VAO_SPHERE[1], prog, T@s, col3, proj, view, eye)

    def _draw_line_seg(p1, p2, col4, mvp):
        pts = np.array([p1,p2],dtype=np.float32)
        _upload_lines(VBO_LINES, pts)
        glUseProgram(prog_flat)
        _u4(prog_flat,'uMVP',mvp)
        _uf4(prog_flat,'uColor',*col4)
        glBindVertexArray(VAO_LINES)
        glDrawArrays(GL_LINES,0,2)
        glBindVertexArray(ctypes.c_uint(0))

    def _draw_joint_frames(link_T, proj, view, eye):
        """Draw RGB axes at each joint origin. Length scales with dist."""
        L = max(0.04, S.cam_dist * 0.025)  # axis length scales with zoom
        mvp = proj @ view
        for jname in joint_order:
            # find the child link of this joint in link_T
            child = jname.replace('_joint', '')
            T = None
            for lnk in link_T:
                if child in lnk and 'visual' not in lnk.lower():
                    T = link_T[lnk]; break
            if T is None:
                for lnk in link_T:
                    if child in lnk:
                        T = link_T[lnk]; break
            if T is None:
                continue
            origin = T[:3, 3]
            # columns of rotation matrix = X,Y,Z axes in world frame
            Rw = T[:3, :3]
            ax = Rw[:, 0]  # X = red
            ay = Rw[:, 1]  # Y = green
            az = Rw[:, 2]  # Z = blue
            _draw_line_seg(origin, origin + ax*L, (1,0,0,0.9), mvp)
            _draw_line_seg(origin, origin + ay*L, (0,1,0,0.9), mvp)
            _draw_line_seg(origin, origin + az*L, (0,0,1,0.9), mvp)
            # draw a small sphere at origin
            _draw_sphere(origin, L*0.12, (0.9,0.9,0.2), prog_lit, proj, view, eye)

    def _draw_ground(proj, view, eye):
        rx = S.cam_target[0]
        ry = S.cam_target[1]
        step = 0.5
        snap = 2 * step
        ox = math.floor(rx/snap)*snap
        oy = math.floor(ry/snap)*snap
        T = np.eye(4,dtype=np.float32)
        T[0,3]=ox; T[1,3]=oy
        # MuJoCo-style checker: medium blue-grey dark, lighter blue-grey light
        _draw_mesh(VAO_GROUND_DARK[0], VAO_GROUND_DARK[1],  prog_lit, T,
                   (0.32,0.38,0.44), proj, view, eye)   # darker tile
        _draw_mesh(VAO_GROUND_LIGHT[0],VAO_GROUND_LIGHT[1], prog_lit, T,
                   (0.50,0.57,0.64), proj, view, eye)   # lighter tile

    def render_3d():
        vw = _vp_w()
        bot = max(80, int(win.height * _layout['bot_frac']))
        glViewport(0,bot,vw,win.height-bot)
        glClearColor(0.52,0.60,0.68,1.0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LEQUAL)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)

        proj,view,eye = _matrices()
        mvp = proj@view

        if S.show_grid:
            _draw_ground(proj,view,eye)

        # Axes
        L=0.4
        for p2,c in [([L,0,0],(1,0,0,1)),([0,L,0],(0,1,0,1)),([0,0,L],(0,0,1,1))]:
            _draw_line_seg([0,0,0],p2,c,mvp)

        if S.data is None: return
        # Use G.frame when user is scrubbing graphs, else S.frame
        # Opaque Robot: ALWAYS S.frame
        q,torso_pos,torso_rpy = S.frame_state(S.frame)
        link_T = fk_fn(q, torso_pos, torso_rpy)

        # Trajectory
        if S.show_trajectory:
            start = max(0, S.frame-S.traj_length)
            for i in range(start, S.frame-1):
                t=(i-start)/max(S.frame-start,1)
                _draw_line_seg(
                    [S.data['torso_x'][i],   S.data['torso_y'][i],   S.data['torso_z'][i]],
                    [S.data['torso_x'][i+1], S.data['torso_y'][i+1], S.data['torso_z'][i+1]],
                    (0.4+0.5*t,0.4+0.5*t,0.1,0.4+0.5*t), mvp)

        # Robot meshes
        for link_name, mesh_key in LINK_TO_MESH.items():
            if link_name not in link_T: continue
            model = link_T[link_name]
            # determine leg color
            found_leg = None
            for lbl in legs:
                if lbl in link_name.upper() or lbl in link_name:
                    found_leg = lbl; break
            col = (0.92, 0.92, 0.94)
            if mesh_key in MESH_VAOS:
                _draw_mesh(MESH_VAOS[mesh_key][0],MESH_VAOS[mesh_key][1],
                           prog_lit, model, col, proj, view, eye)

        # ── Ghost robot: drawn at S.frame when G.frame != S.frame ────────────
        # Alpha ramps from 0 (no offset) to 0.38 (max offset = ±50 frames)
        ghost_alpha = 0.25 if G.play_mode == 'main' else 0.38
        if ghost_alpha > 0.01:
            gq, gpos, grpy = S.frame_state(G.frame)
            ghost_T = fk_fn(gq, gpos, grpy)
            for link_name, mesh_key in LINK_TO_MESH.items():
                if link_name not in ghost_T: continue
                if mesh_key not in MESH_VAOS: continue
                # Ghost is desaturated white-blue tint
                _draw_mesh(MESH_VAOS[mesh_key][0], MESH_VAOS[mesh_key][1],
                           prog_lit, ghost_T[link_name],
                           (0.75, 0.82, 1.0),   # cool white-blue tint
                           proj, view, eye, alpha=ghost_alpha)

        # Foot spheres + force arrows + limit warnings
        # Foot order matches robot.py feet_names: [FR, FL, RR, RL]
        leg_keys = list(legs.keys())   # e.g. ['FR','FL','RR','RL']
        has_foot_pos = ('foot_pos' in S.data and
                        S.data['foot_pos'] is not None and
                        np.any(S.data['foot_pos'] != 0))
        for lbl, ji in legs.items():
            fi2 = leg_keys.index(lbl) if lbl in leg_keys else 0

            # ── Foot contact position ────────────────────────────────────────
            # Prefer data-driven world-frame position (from robot._foot_positions)
            # Fall back to FK link position if file not present
            if has_foot_pos:
                foot_pos = S.data['foot_pos'][S.frame, fi2*3:(fi2+1)*3].astype(np.float32)
            else:
                foot_link = None
                for link_name in link_T:
                    low = link_name.lower()
                    if lbl.lower() in low and 'foot' in low:
                        foot_link = link_name; break
                foot_pos = link_T[foot_link][:3,3] if (foot_link and foot_link in link_T) else None

            if foot_pos is None:
                continue

            # ── Contact sphere ───────────────────────────────────────────────
            cv   = S.data.get(f'contact_{lbl}')
            in_c = cv is not None and cv[S.frame] > 0.5
            fc   = (0.1,1.0,0.4) if (S.show_contacts and in_c) else (0.15,0.15,0.15)
            _draw_sphere(foot_pos, 0.025, fc, prog_lit, proj, view, eye)

            # ── Force arrow from foot contact point ──────────────────────────
            # fv = [Fx, Fy, Fz] in world frame; arrow starts at foot contact pos
            if S.show_forces:
                fv = S.data['foot_forces'][S.frame, fi2*3:(fi2+1)*3]
                mag = np.linalg.norm(fv)
                if mag > 1.0:
                    tip = foot_pos + fv * S.force_scale
                    # Color by leg for readability
                    leg_col = get_leg_color_f(lbl)
                    _draw_line_seg(foot_pos, tip, (leg_col[0],leg_col[1],leg_col[2],0.9), mvp)

            # joint limit warnings
            if S.show_limits:
                for i in ji:
                    jname = joint_order[i]
                    lo,hi = joint_limits[jname]
                    if is_near_limit(lo,hi,float(q[i])):
                        # find link transform for this joint
                        for link_name in link_T:
                            if jname.replace('_joint','') in link_name:
                                pos = link_T[link_name][:3,3]
                                _draw_sphere(pos,0.028,(1,0.1,0.1),
                                             prog_lit,proj,view,eye)
                                break


        # Joint frame axes (F6 to toggle, F7 to toggle limit warnings)
        if S.show_joint_frames:
            _draw_joint_frames(link_T, proj, view, eye)
    # ── ImGui panel ───────────────────────────────────────────────────────────

    # ── Graph system ──────────────────────────────────────────────────────────
    #
    # Each "graph" contains multiple "series" (lines).
    # A graph has shared Y bounds (lo/hi), shared window.
    # Each series has: param key, color, enabled checkbox, label.
    # Lines turn RED where value is out of [lo, hi].
    # Hovering a series highlights it and dims others.

    _GRAPH_PARAMS = {}

    # Scalars
    _GRAPH_PARAMS['torso_x']      = lambda d,f: float(d['torso_x'][f])
    _GRAPH_PARAMS['torso_y']      = lambda d,f: float(d['torso_y'][f])
    _GRAPH_PARAMS['torso_z']      = lambda d,f: float(d['torso_z'][f])
    _GRAPH_PARAMS['torso_roll']   = lambda d,f: math.degrees(float(d['torso_roll'][f]))
    _GRAPH_PARAMS['torso_pitch']  = lambda d,f: math.degrees(float(d['torso_pitch'][f]))
    _GRAPH_PARAMS['torso_yaw']    = lambda d,f: math.degrees(float(d['torso_yaw'][f]))
    _GRAPH_PARAMS['desired_vel_x']= lambda d,f: float(d['desired_vel_x'][f])
    _GRAPH_PARAMS['contact_FL']   = lambda d,f: float(d['contact_FL'][f]) if d['contact_FL'] is not None else 0.0
    _GRAPH_PARAMS['contact_FR']   = lambda d,f: float(d['contact_FR'][f]) if d['contact_FR'] is not None else 0.0
    _GRAPH_PARAMS['contact_RL']   = lambda d,f: float(d['contact_RL'][f]) if d['contact_RL'] is not None else 0.0
    _GRAPH_PARAMS['contact_RR']   = lambda d,f: float(d['contact_RR'][f]) if d['contact_RR'] is not None else 0.0
    for _ji, _jn in enumerate(joint_order):
        _GRAPH_PARAMS[f'q {_jn}']   = (lambda j: lambda d,f: math.degrees(float(d['q'][f,j])))(_ji)
        _GRAPH_PARAMS[f'tau {_jn}'] = (lambda j: lambda d,f: float(d['tau'][f,j]))(_ji)
        _GRAPH_PARAMS[f'dq {_jn}']  = (lambda j: lambda d,f: float(d['dq'][f,j]))(_ji)

    _PARAM_KEYS = list(_GRAPH_PARAMS.keys())

    # Series colors (cycle through these)
    _SERIES_COLORS = [
        (0.25, 0.55, 1.00),  # blue
        (0.20, 0.85, 0.40),  # green
        (1.00, 0.70, 0.15),  # orange
        (0.85, 0.25, 0.85),  # purple
        (0.15, 0.85, 0.85),  # cyan
        (1.00, 0.35, 0.35),  # red
        (0.85, 0.85, 0.20),  # yellow
        (0.60, 0.40, 1.00),  # violet
    ]

    def _mk_series(param, color_idx, enabled=True, lim_lo=-30.0, lim_hi=30.0):
        r,g,b = _SERIES_COLORS[color_idx % len(_SERIES_COLORS)]
        return {
            'param':    param,
            'enabled':  enabled,
            'color':    (r,g,b),
            'limit_lo': lim_lo,
            'limit_hi': lim_hi,
        }

    def _mk_graph(title, series_list, lo, hi, lo_str, hi_str):
        return {
            'title':    title,
            'series':   series_list,
            'lo':       lo,  'hi':       hi,
            'lo_str':   lo_str, 'hi_str': hi_str,
            'lo_edit':  False,  'hi_edit': False,
            'hovered':  -1,      # index of hovered series (-1 = none)
        }

    # Default preloaded graphs
    def _build_default_graphs():
        graphs = []
        # 1. All joint angles
        jq = [_mk_series(f'q {jn}', i) for i,jn in enumerate(joint_order)]
        graphs.append(_mk_graph('Joint Angles (deg)', jq,
                                -180.0, 180.0, '-180', '180'))
        # 2. RPY
        rpy = [_mk_series('torso_roll',  0),
               _mk_series('torso_pitch', 1),
               _mk_series('torso_yaw',   2)]
        graphs.append(_mk_graph('Body RPY (deg)', rpy,
                                -40.0, 40.0, '-40', '40'))
        # 3. All torques
        jtau = [_mk_series(f'tau {jn}', i) for i,jn in enumerate(joint_order)]
        graphs.append(_mk_graph('Joint Torques (Nm)', jtau,
                                -50.0, 50.0, '-50', '50'))
        return graphs

    _graphs = _build_default_graphs()
    _GRAPH_WINDOW = 50    # half-window: ±50 frames around playhead

    # ── Resizable layout state ──────────────────────────────────────────────
    _layout = {
        'vp_frac':  0.54,   # fraction of window width for 3D viewport
        'bot_frac': 0.13,   # fraction of window height for bottom bar
        'drag_v':   False,  # dragging vertical splitter
        'drag_h':   False,  # dragging horizontal splitter (right side)
        'drag_hL':  False,  # dragging horizontal splitter (viewport+speed)
    }
    _SPLITTER_W = 6  # splitter grab width in pixels


    def _get_val(d, frame, param):
        fn = _GRAPH_PARAMS.get(param)
        if fn is None: return 0.0
        try:    return fn(d, frame)
        except: return 0.0

    # ── layout save/load ─────────────────────────────────────────────────────
    import json as _json
    _LAYOUT_FILE = 'viewer_layout.json'

    def _save_layout():
        data = {
            'vp_frac':   _layout['vp_frac'],
            'bot_frac':  _layout['bot_frac'],
            'win_w':     win.width,
            'win_h':     win.height,
        }
        try:
            with open(_LAYOUT_FILE,'w') as f: _json.dump(data,f,indent=2)
            print(f"Layout saved to {_LAYOUT_FILE}")
        except Exception as ex: print(f"Save failed: {ex}")

    def _load_layout():
        try:
            with open(_LAYOUT_FILE,'r') as f: data=_json.load(f)
            _layout['vp_frac']  = float(data.get('vp_frac',  0.54))
            _layout['bot_frac'] = float(data.get('bot_frac', 0.17))
            w = int(data.get('win_w', win.width))
            h = int(data.get('win_h', win.height))
            win.set_size(w, h)
            print(f"Layout loaded from {_LAYOUT_FILE}  win={w}x{h}")
        except FileNotFoundError: print(f"No layout file found at {_LAYOUT_FILE}")
        except Exception as ex:   print(f"Load failed: {ex}")

    # ── graph playhead drag state ─────────────────────────────────────────────
    _graph_ph_drag  = {}   # gi -> bool: is playhead being dragged
    _graph_center   = {}   # gi -> drag start mouse x
    _graph_offset     = {}   # gi -> int frame offset (unused, kept for compat)
    _graph_view_frame = {}   # gi -> int: local seek-bar frame (default = S.frame)
    _pending_reload = [False]
    _pinned_tips = []

    def _u32(r,g,b,a=1.0):
        return imgui.get_color_u32_rgba(r,g,b,a)

    def _short_label(param):
        lbl = param.split(' ',1)[-1] if ' ' in param else param
        lbl = lbl.replace('_hip_joint','_h').replace('_thigh_joint','_t').replace('_calf_joint','_c')
        return lbl.replace('_joint','').replace('_','-')

    def _render_graphs(imgui, S):
        d = S.data
        if d is None: return

        # Clean up pinned tips out of window
        _pinned_tips[:] = [t for t in _pinned_tips if abs(t['frame'] - S.frame) <= _GRAPH_WINDOW]

        # Save/load + reload robot buttons
        if imgui.button('Save Layout##sl'): _save_layout()
        imgui.same_line(spacing=6)
        if imgui.button('Load Layout##ll'): _load_layout()
        imgui.same_line(spacing=6)
        if imgui.button('Load Robot##lr'):
            _pending_reload[0] = True
        imgui.separator()

        pw    = imgui.get_content_region_available_width()
        gh    = 160
        # Side panel is wide enough for 30-char labels
        SIDE_W = 200

        gi = 0
        while gi < len(_graphs):
            graph  = _graphs[gi]
            lo, hi = graph['lo'], graph['hi']
            series = graph['series']
            span   = hi - lo if hi != lo else 1.0

            # ── header row ────────────────────────────────────────────────────
            imgui.text_colored(graph['title'][:30], 0.65,0.75,0.88)
            imgui.same_line(spacing=6)
            imgui.text_disabled('[')
            imgui.same_line(spacing=1)
            imgui.push_item_width(54)
            if graph['lo_edit']:
                ch,ns = imgui.input_text(f'##lo{gi}', graph['lo_str'], 14,
                                         imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                if ch:
                    try: graph['lo']=float(ns); graph['lo_str']=ns
                    except: pass
                    graph['lo_edit'] = False
                if not imgui.is_item_active(): graph['lo_edit'] = False
            else:
                any_lo = any(_get_val(d,G.frame,s['param'])<lo for s in series if s['enabled'])
                imgui.text_colored(f'{lo:.3g}', *(1.0,0.25,0.25) if any_lo else (0.5,0.5,0.5))
                if imgui.is_item_clicked(): graph['lo_edit']=True
            imgui.pop_item_width()
            imgui.same_line(spacing=1); imgui.text_disabled(','); imgui.same_line(spacing=1)
            imgui.push_item_width(54)
            if graph['hi_edit']:
                ch,ns = imgui.input_text(f'##hi{gi}', graph['hi_str'], 14,
                                         imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                if ch:
                    try: graph['hi']=float(ns); graph['hi_str']=ns
                    except: pass
                    graph['hi_edit'] = False
                if not imgui.is_item_active(): graph['hi_edit'] = False
            else:
                any_hi = any(_get_val(d,S.frame,s['param'])>hi for s in series if s['enabled'])
                imgui.text_colored(f'{hi:.3g}', *(1.0,0.25,0.25) if any_hi else (0.5,0.5,0.5))
                if imgui.is_item_clicked(): graph['hi_edit']=True
            imgui.pop_item_width()
            imgui.same_line(spacing=1); imgui.text_disabled(']')
            imgui.same_line(spacing=8)
            if imgui.button(f'X##xg{gi}', 20, 0):
                _graphs.pop(gi); continue

            gi += 1
            if gi > len(_graphs): break
            graph  = _graphs[gi-1]
            lo, hi = graph['lo'], graph['hi']
            series = graph['series']
            span   = hi - lo if hi != lo else 1.0
            gi2    = gi - 1

            body_start_x, body_start_y = imgui.get_cursor_screen_pos()

            # ── side panel ────────────────────────────────────────────────────
            imgui.push_style_color(getattr(imgui,'COLOR_CHILD_BACKGROUND',8), 0.07,0.07,0.10,1.0)
            in_child = False
            try:    in_child = imgui.begin_child(f'##side{gi}', SIDE_W, gh, border=True)
            except:
                try: in_child = imgui.begin_child(f'##side{gi}', SIDE_W, gh)
                except: pass

            if in_child:
                # '+' opens a popup to choose param — NOT immediate add
                if imgui.button(f'+##as{gi}', 22, 0):
                    imgui.open_popup(f'##addpop{gi}')

                if imgui.begin_popup(f'##addpop{gi}'):
                    imgui.text('Add series:')
                    imgui.separator()
                    for pk in _PARAM_KEYS:
                        lbl = _short_label(pk)[:28]
                        if imgui.selectable(f'{lbl}##{gi}_{pk}')[0]:
                            ci = len(series)
                            series.append(_mk_series(pk, ci))
                            imgui.close_current_popup()
                    imgui.end_popup()

                imgui.separator()

                # Series list (newest first so delete is safe)
                for si in range(len(series)-1, -1, -1):
                    s = series[si]
                    r2,g2,b2 = s['color']
                    # colored checkbox
                    try:
                        imgui.push_style_color(getattr(imgui,'COLOR_CHECK_MARK',27), r2,g2,b2,1.0)
                        _, s['enabled'] = imgui.checkbox(f'##{gi}_{si}c', s['enabled'])
                        imgui.pop_style_color()
                    except:
                        _, s['enabled'] = imgui.checkbox(f'##{gi}_{si}c', s['enabled'])
                    imgui.same_line(spacing=3)
                    # param label (show name, click to change via combo popup)
                    lbl = _short_label(s['param'])[:18]
                    imgui.text_colored(f'{lbl}', r2,g2,b2)
                    # inline mini combo on click
                    if imgui.is_item_clicked():
                        series[si]['_edit'] = True
                    if series[si].get('_edit', False):
                        imgui.push_item_width(SIDE_W - 6)
                        cur_idx = _PARAM_KEYS.index(s['param']) if s['param'] in _PARAM_KEYS else 0
                        try:
                            ch2, ni = imgui.combo(f'##{gi}_{si}p', cur_idx, _PARAM_KEYS)
                            if ch2:
                                s['param'] = _PARAM_KEYS[ni]
                                series[si]['_edit'] = False
                        except: pass
                        imgui.pop_item_width()
                        if not imgui.is_item_active() and not imgui.is_item_focused():
                            series[si]['_edit'] = False
                    imgui.same_line(spacing=4)
                    # Bound input text boxes
                    imgui.push_item_width(38)
                    ch_lo, new_lo_s = imgui.input_text(
                        f'##lo{gi}_{si}', s.get('lim_lo_str', str(int(s['limit_lo']))), 8,
                        imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                    if ch_lo:
                        try: s['limit_lo']=float(new_lo_s); s['lim_lo_str']=new_lo_s
                        except ValueError: pass
                    imgui.pop_item_width()
                    imgui.same_line(spacing=2)
                    imgui.push_item_width(38)
                    ch_hi, new_hi_s = imgui.input_text(
                        f'##hi{gi}_{si}', s.get('lim_hi_str', str(int(s['limit_hi']))), 8,
                        imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                    if ch_hi:
                        try: s['limit_hi']=float(new_hi_s); s['lim_hi_str']=new_hi_s
                        except ValueError: pass
                    imgui.pop_item_width()
                    imgui.same_line(spacing=2)
                    if imgui.button(f'-##{gi}_{si}r', 20, 0):
                        series.pop(si); continue
                    # current value, red if outside per-series bounds
                    cv = _get_val(d, G.frame, s['param'])
                    oob = cv < s['limit_lo'] or cv > s['limit_hi']
                    imgui.text_colored(f'  {cv:+.3f}', *(1.0,0.2,0.2) if oob else (r2*0.85,g2*0.85,b2*0.85))

            imgui.end_child()
            imgui.pop_style_color()

            # ── graph canvas ──────────────────────────────────────────────────
            imgui.same_line(spacing=2)
            canvas_w = max(60, pw - SIDE_W - 6)

            # Capture cursor position at top-left of canvas area (before any drawing)
            canvas_origin_x, canvas_origin_y = imgui.get_cursor_screen_pos()
            dl = imgui.get_window_draw_list()
            sk_h = 14

            # ── Seek bar: shows only the ±50 window, click moves LOCAL view only ─
            sk_x  = canvas_origin_x
            sk_y  = canvas_origin_y
            # window bounds for this graph (same as the graph canvas below)
            sk_start = max(0, S.frame - _GRAPH_WINDOW)
            sk_end   = min(S.n_frames, S.frame + _GRAPH_WINDOW)
            sk_span  = max(sk_end - sk_start, 1)
            # G.frame is the shared graph scrub position
            gf = G.frame

            # Background
            dl.add_rect_filled(sk_x, sk_y, sk_x+canvas_w, sk_y+sk_h, _u32(0.07,0.07,0.10))
            # Tick marks within ±50 window
            for ti in range(0, sk_span+1, max(1, sk_span//8)):
                tx  = sk_x + canvas_w * ti / sk_span
                fi_t= max(0,min(S.n_frames-1, sk_start+ti))
                dl.add_line(tx, sk_y+sk_h-5, tx, sk_y+sk_h, _u32(0.3,0.3,0.35,0.6), 1)
                if d is not None and d.get('time') is not None:
                    try:    tl = f'{float(d["time"][fi_t]):.2f}s'
                    except: tl = f'{fi_t-S.frame:+d}'
                else:
                    tl = f'{fi_t-S.frame:+d}'
                if ti < sk_span - sk_span//10:
                    dl.add_text(tx+2, sk_y+1, _u32(0.32,0.32,0.38), tl)
            # Main timeline (S.frame) — white, always in center of window
            ph_sk_x = sk_x + canvas_w * (S.frame - sk_start) / sk_span
            dl.add_line(ph_sk_x, sk_y, ph_sk_x, sk_y+sk_h, _u32(0.8,0.8,1.0,0.9), 2)
            dl.add_triangle_filled(ph_sk_x-4,sk_y, ph_sk_x+4,sk_y, ph_sk_x,sk_y+7, _u32(0.8,0.8,1.0,1.0))
            # Graph scrub (G.frame) — yellow
            if sk_start <= gf <= sk_end:
                gf_sk_x = sk_x + canvas_w * (gf - sk_start) / sk_span
                dl.add_line(gf_sk_x, sk_y, gf_sk_x, sk_y+sk_h, _u32(1,0.92,0.2,0.8), 1.5)
                dl.add_triangle_filled(gf_sk_x-4,sk_y+sk_h, gf_sk_x+4,sk_y+sk_h, gf_sk_x,sk_y+sk_h-7, _u32(1,0.92,0.2,0.9))

            # Click/drag seek bar → G.scrub() — NEVER moves S.frame
            imgui.invisible_button(f'##sk{gi2}', canvas_w, sk_h)
            if imgui.is_item_hovered() and imgui.is_mouse_down(0):
                mx_sk = imgui.get_mouse_pos().x
                new_gf = sk_start + int((mx_sk - sk_x) / canvas_w * sk_span)
                G.scrub(new_gf, S.n_frames)
            elif not imgui.is_mouse_down(0) and _graph_ph_drag.get(f'sk{gi2}', False):
                _graph_ph_drag[f'sk{gi2}'] = False
                # G.frame stays where it is — no release/reset
            if imgui.is_item_hovered() and imgui.is_mouse_down(0):
                _graph_ph_drag[f'sk{gi2}'] = True

            # ── Graph canvas starts immediately after seek bar ─────────────────
            gx  = canvas_origin_x
            gy2 = canvas_origin_y + sk_h

            dl.add_rect_filled(gx, gy2, gx+canvas_w, gy2+gh, _u32(0.05,0.05,0.07))
            dl.add_rect(gx, gy2, gx+canvas_w, gy2+gh, _u32(0.18,0.18,0.22), 0, 1.0)

            # Clip drawing to graph bounds
            dl.push_clip_rect(gx, gy2, gx+canvas_w, gy2+gh, True)

            def vy(v): return gy2 + gh - (v-lo)/span*gh

            if lo < 0 < hi:
                dl.add_line(gx, vy(0), gx+canvas_w, vy(0), _u32(0.32,0.32,0.32,0.3), 1)
            # Graph canvas: window anchored to S.frame ±50
            # G.frame = shared graph scrub (yellow), S.frame = main (white)
            start = max(0, S.frame - _GRAPH_WINDOW)
            end   = min(S.n_frames, S.frame + _GRAPH_WINDOW)
            nw    = max(end - start, 2)
            g_center  = max(start, min(end-1, G.frame))
            ph_x      = gx + canvas_w * (g_center - start) / max(nw-1,1)
            main_ph_x = gx + canvas_w * (S.frame  - start) / max(nw-1,1)
            dl.add_rect_filled(ph_x, gy2, gx+canvas_w, gy2+gh, _u32(1,1,1,0.025))

            # hover
            mx2, my2 = imgui.get_mouse_pos()
            mouse_in  = gx <= mx2 <= gx+canvas_w and gy2 <= my2 <= gy2+gh
            hov_si    = -1
            if mouse_in:
                bd = 18.0
                for si2, s in enumerate(series):
                    if not s['enabled']: continue
                    fn = _GRAPH_PARAMS.get(s['param'])
                    if fn is None: continue
                    fi_m = start + int((mx2-gx)/canvas_w*(nw-1))
                    fi_m = max(start, min(end-1, fi_m))
                    try:
                        pd = abs(vy(fn(d, fi_m)) - my2)
                        if pd < bd: bd=pd; hov_si=si2
                    except: pass
            graph['hovered'] = hov_si

            # draw lines — red when outside per-series bounds
            for si2, s in enumerate(series):
                if not s['enabled']: continue
                fn = _GRAPH_PARAMS.get(s['param'])
                if fn is None: continue
                r2,g2,b2 = s['color']
                s_lo = s['limit_lo']
                s_hi = s['limit_hi']
                is_hov    = (hov_si == si2)
                is_dimmed = (hov_si >= 0 and not is_hov)
                base_a    = 0.20 if is_dimmed else (1.0 if is_hov else 0.85)
                lw        = 2.5 if is_hov else 1.4

                # Bound lines + labels when hovered
                if is_hov and mouse_in:
                    dl.add_line(gx, vy(s_lo), gx+canvas_w, vy(s_lo), _u32(r2,g2,b2,0.6), 1)
                    dl.add_line(gx, vy(s_hi), gx+canvas_w, vy(s_hi), _u32(r2,g2,b2,0.6), 1)
                    dl.add_text(gx+canvas_w-44, vy(s_hi)-12, _u32(r2,g2,b2,0.85), f'{s_hi:.4g}')
                    dl.add_text(gx+canvas_w-44, vy(s_lo)+2,  _u32(r2,g2,b2,0.85), f'{s_lo:.4g}')

                pp3=ppy3=pv2=None
                for k in range(nw):
                    fi2 = start+k
                    try: v2 = fn(d,fi2)
                    except: v2 = 0.0
                    px4 = gx + canvas_w*k/(nw-1)
                    py4 = vy(v2)
                    if pp3 is not None:
                        fut  = fi2 > G.frame
                        a2   = base_a*(0.35 if fut else 1.0)
                        c_in  = _u32(r2,g2,b2,a2)
                        c_out = _u32(1.0,0.15,0.15,a2)
                        # Find all crossing t-values for s_lo and s_hi
                        ts = [0.0, 1.0]
                        dv = v2 - pv2
                        if abs(dv) > 1e-9:
                            t_lo = (s_lo - pv2) / dv
                            if 0.0 < t_lo < 1.0: ts.append(t_lo)
                            t_hi = (s_hi - pv2) / dv
                            if 0.0 < t_hi < 1.0: ts.append(t_hi)
                        ts.sort()
                        for ti in range(len(ts)-1):
                            ta, tb = ts[ti], ts[ti+1]
                            xa = pp3  + (px4-pp3)*ta;  ya = ppy3 + (py4-ppy3)*ta
                            xb = pp3  + (px4-pp3)*tb;  yb = ppy3 + (py4-ppy3)*tb
                            vm = pv2 + dv * (ta+tb)*0.5   # midpoint value
                            col2 = c_out if (vm < s_lo or vm > s_hi) else c_in
                            dl.add_line(xa, ya, xb, yb, col2, lw)
                    pp3,ppy3,pv2 = px4,py4,v2

            dl.pop_clip_rect()

            # Main S.frame: white vertical bar (always at center of window)
            dl.add_line(main_ph_x, gy2, main_ph_x, gy2+gh, _u32(0.8,0.8,1.0,0.55), 1.5)
            # Local view playhead (yellow, draggable triangle)
            dl.add_line(ph_x, gy2, ph_x, gy2+gh, _u32(1,0.92,0.2,0.9), 1.5)
            hw=8; th=12
            dl.add_triangle_filled(ph_x-hw,gy2,    ph_x+hw,gy2,    ph_x,gy2+th,  _u32(1,0.92,0.2,1.0))
            dl.add_triangle_filled(ph_x-hw,gy2+gh, ph_x+hw,gy2+gh, ph_x,gy2+gh-th,_u32(1,0.92,0.2,0.75))

            # Drag yellow triangle → G.scrub() — all graphs sync, S.frame untouched
            is_ph_drag = _graph_ph_drag.get(gi2, False)
            near_ph    = abs(mx2-ph_x) < hw+6
            if imgui.is_mouse_down(0):
                if is_ph_drag or (mouse_in and near_ph):
                    _graph_ph_drag[gi2] = True
                    fi_new = start + int((mx2-gx)/canvas_w*(nw-1))
                    G.scrub(fi_new, S.n_frames)
            else:
                if is_ph_drag:
                    _graph_ph_drag[gi2] = False
                    # G.frame stays — no reset on release

            imgui.invisible_button(f'##gc{gi2}', canvas_w, gh)

            # ── X axis: labels in ±50 window coords ──────────────────────────
            axis_h = 16
            ax_y   = gy2 + gh
            dl.add_rect_filled(gx, ax_y, gx+canvas_w, ax_y+axis_h, _u32(0.04,0.04,0.06))
            n_ticks = 6
            for ti in range(n_ticks+1):
                frac = ti / n_ticks
                fi_t = start + int(frac * (nw-1))
                fi_t = max(0, min(S.n_frames-1, fi_t))
                tx   = gx + canvas_w * frac
                dl.add_line(tx, ax_y, tx, ax_y+4, _u32(0.35,0.35,0.38,0.8), 1)
                # relative frame offset from S.frame
                rel = fi_t - S.frame
                if d is not None and d.get('time') is not None:
                    try:    lbl = f'{float(d["time"][fi_t]):.2f}s'
                    except: lbl = f'{rel:+d}'
                else:
                    lbl = f'{rel:+d}'
                if ti < n_ticks:
                    dl.add_text(tx+2, ax_y+4, _u32(0.38,0.38,0.4), lbl)
            # cursor readout at mouse position
            fi_cur_t = start + int((mx2-gx)/canvas_w*(nw-1)) if mouse_in else G.frame
            fi_cur_t = max(start, min(end-1, fi_cur_t))
            rel_cur  = fi_cur_t - S.frame
            if d is not None and d.get('time') is not None:
                try:    cur_lbl = f'f{fi_cur_t} ({rel_cur:+d}) {float(d["time"][fi_cur_t]):.3f}s'
                except: cur_lbl = f'f{fi_cur_t} ({rel_cur:+d})'
            else:
                cur_lbl = f'f{fi_cur_t} ({rel_cur:+d})'
            dl.add_text(max(gx+2, min(main_ph_x+3, gx+canvas_w-80)), ax_y+4,
                        _u32(0.85,0.85,1.0,0.85), cur_lbl)

            imgui.set_cursor_screen_pos((body_start_x, body_start_y + sk_h + gh + axis_h + 6))
            imgui.spacing()

        # ── add graph button ──────────────────────────────────────────────────
        imgui.separator()
        if imgui.button('+ New Graph##ng'):
            imgui.open_popup('##newgraph_pop')
        if imgui.begin_popup('##newgraph_pop'):
            imgui.text('Choose first series:')
            imgui.separator()
            for pk in _PARAM_KEYS:
                lbl = _short_label(pk)[:28]
                if imgui.selectable(f'{lbl}##ngp_{pk}')[0]:
                    title = _short_label(pk)[:30]
                    _graphs.append(_mk_graph(title,
                                   [_mk_series(pk, 0)],
                                   -1.0,1.0,'-1.0','1.0'))
                    imgui.close_current_popup()
            imgui.end_popup()


    _tl_drag = {'active': None}

    def _timeline_bar(dl, px,py,bw,bh, n_fr,lo,hi,cur):
        """All parts (IN strip, OUT strip, main bar) drawn INSIDE [py, py+bh]."""
        MINI_H = 16
        MAIN_H = max(20, bh - MINI_H*2 - 2)
        in_y   = py
        out_y  = py + MINI_H + 1
        main_y = py + MINI_H*2 + 2

        xlo = px + bw*lo/max(n_fr-1,1)
        xhi = px + bw*hi/max(n_fr-1,1)
        xfr = px + bw*cur/max(n_fr-1,1)

        # IN strip (green)
        dl.add_rect_filled(px, in_y, px+bw, in_y+MINI_H, _u32(0.07,0.11,0.07))
        dl.add_rect_filled(xlo, in_y, px+bw, in_y+MINI_H, _u32(0.08,0.28,0.08,0.55))
        dl.add_line(xlo, in_y, xlo, in_y+MINI_H, _u32(0.2,0.9,0.2,0.9), 2)
        dl.add_triangle_filled(xlo-4,in_y, xlo+4,in_y, xlo,in_y+MINI_H, _u32(0.2,0.9,0.2,1.0))
        dl.add_line(xfr, in_y, xfr, in_y+MINI_H, _u32(1,0.92,0.2,0.4), 1)
        dl.add_text(px+2, in_y, _u32(0.25,0.75,0.25,0.9), f'IN {lo}')

        # OUT strip (red)
        dl.add_rect_filled(px, out_y, px+bw, out_y+MINI_H, _u32(0.11,0.07,0.07))
        dl.add_rect_filled(px, out_y, xhi, out_y+MINI_H, _u32(0.28,0.08,0.08,0.55))
        dl.add_line(xhi, out_y, xhi, out_y+MINI_H, _u32(0.9,0.2,0.2,0.9), 2)
        dl.add_triangle_filled(xhi-4,out_y, xhi+4,out_y, xhi,out_y+MINI_H, _u32(0.9,0.2,0.2,1.0))
        dl.add_line(xfr, out_y, xfr, out_y+MINI_H, _u32(1,0.92,0.2,0.4), 1)
        dl.add_text(px+bw-52, out_y, _u32(0.75,0.25,0.25,0.9), f'OUT {hi}')

        # Main bar
        dl.add_rect_filled(px,main_y,px+bw,main_y+MAIN_H, _u32(0.09,0.09,0.11))
        dl.add_rect_filled(xlo,main_y,xhi,main_y+MAIN_H, _u32(0.12,0.40,0.12,0.38))
        dl.add_rect(xlo,main_y,xhi,main_y+MAIN_H, _u32(0.2,0.65,0.2,0.5),0,1.0)
        # ±50 window band
        w50_lo = max(0, cur-_GRAPH_WINDOW)
        w50_hi = min(n_fr-1, cur+_GRAPH_WINDOW)
        xwl = px + bw*w50_lo/max(n_fr-1,1)
        xwh = px + bw*w50_hi/max(n_fr-1,1)
        mid  = main_y + MAIN_H//2
        dl.add_rect_filled(xwl,mid-2,xwh,mid+2, _u32(0.4,0.6,1.0,0.35))
        dl.add_line(xwl,main_y,xwl,main_y+MAIN_H, _u32(0.4,0.6,1.0,0.45),1)
        dl.add_line(xwh,main_y,xwh,main_y+MAIN_H, _u32(0.4,0.6,1.0,0.45),1)
        # Tick marks
        for i in range(0,11):
            tx=px+bw*i/10
            dl.add_line(tx,main_y+MAIN_H-5,tx,main_y+MAIN_H,_u32(0.35,0.35,0.35,0.6),1)
            dl.add_text(tx+2,main_y+MAIN_H-14,_u32(0.42,0.42,0.42), str(int(n_fr*i/10)))
        # Playhead
        dl.add_line(xfr,main_y,xfr,main_y+MAIN_H, _u32(1,0.92,0.2,1),2)
        hw=8; th=min(12, MAIN_H//2)
        dl.add_triangle_filled(xfr-hw,main_y,    xfr+hw,main_y,    xfr,main_y+th,      _u32(1,0.92,0.2,1))
        dl.add_triangle_filled(xfr-hw,main_y+MAIN_H, xfr+hw,main_y+MAIN_H, xfr,main_y+MAIN_H-th, _u32(1,0.92,0.2,0.8))

        # Interaction — all zones within [py, py+bh]
        mx,my = imgui.get_mouse_pos()
        in_widget = (px<=mx<=px+bw) and (py<=my<=py+bh)
        new_lo,new_hi,new_cur=lo,hi,cur; changed=False
        if imgui.is_mouse_down(0):
            if _tl_drag['active'] is None and in_widget:
                if   my <= in_y+MINI_H:   _tl_drag['active']= 'in'
                elif my <= out_y+MINI_H:  _tl_drag['active']= 'out'
                elif abs(mx-xlo)<10:      _tl_drag['active']= 'in'
                elif abs(mx-xhi)<10:      _tl_drag['active']= 'out'
                else:                     _tl_drag['active']= 'seek'
            a=_tl_drag['active']
            if a=='in':
                new_lo=int((mx-px)/bw*(n_fr-1)); new_lo=max(0,min(hi-1,new_lo)); changed=True
            elif a=='out':
                new_hi=int((mx-px)/bw*(n_fr-1)); new_hi=max(lo+1,min(n_fr-1,new_hi)); changed=True
            elif a=='seek':
                new_cur=int((mx-px)/bw*(n_fr-1)); new_cur=max(lo,min(hi,new_cur)); changed=True
        else:
            _tl_drag['active']=None
        return new_lo,new_hi,new_cur,changed

    def render_panel():
        _io.display_size=(win.width,win.height)
        try: _io.mouse_pos=(_mx, win.height-_my)
        except Exception: pass
        try:
            _io.mouse_down[0]=_mb[0]
            _io.mouse_down[1]=_mb[1]
            _io.mouse_down[2]=_mb[2]
        except Exception: pass
        imgui.new_frame()

        vp_w   = _vp_w()
        gp_w   = win.width - vp_w
        bot_h  = max(80, int(win.height * _layout['bot_frac']))
        top_h  = win.height - bot_h
        flags  = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
                  imgui.WINDOW_NO_MOVE     | imgui.WINDOW_NO_COLLAPSE)

        # ══════════════════════════════════════════════════════════════════════
        # RIGHT: Live Graphs Panel — no scrollbar on the outer window,
        # graphs scroll internally via child windows
        # ══════════════════════════════════════════════════════════════════════
        SET_H  = 120
        GRAPH_H = top_h - SET_H - 4

        # ── Graphs panel (top portion of right side) ──────────────────────────
        imgui.set_next_window_position(vp_w, 0)
        imgui.set_next_window_size(gp_w, top_h)
        imgui.begin('##graphs', flags | imgui.WINDOW_NO_SCROLLBAR)

        imgui.text_colored('LIVE GRAPHS', 0.55,0.75,1.0)
        imgui.separator()

        # Reserve SET_H pixels at the bottom for settings; give the rest to graphs
        HEADER_H = 42   # approx height of title + separator
        scroll_h = max(60, top_h - HEADER_H - SET_H)
        try:
            imgui.begin_child('##gscroll', 0, scroll_h, border=False)
        except:
            pass
        if S.data is not None:
            _render_graphs(imgui, S)
        try:
            imgui.end_child()
        except:
            pass
        # ── Settings: fixed section, always visible ─────────────────────────
        imgui.separator()
        imgui.text_colored('SETTINGS', 0.65,0.75,0.88)
        imgui.separator()
        imgui.text_colored('CAM', 0.5,0.6,0.75); imgui.same_line(spacing=6)
        if imgui.radio_button('Main##fm', S.cam_follow_mode == 'main'):
            S.cam_follow_mode = 'main'; S.update_cam()
        imgui.same_line(spacing=4)
        if imgui.radio_button('Ghost##fg', S.cam_follow_mode == 'ghost'):
            S.cam_follow_mode = 'ghost'; S.update_cam()
        imgui.same_line(spacing=8)
        imgui.push_item_width(62)
        cv,vv=imgui.drag_float('##cy2',S.cam_yaw,   0.5,-180,180,'Y%.0f')
        if cv: S.cam_yaw=vv
        imgui.pop_item_width(); imgui.same_line(spacing=2)
        imgui.push_item_width(62)
        cv,vv=imgui.drag_float('##cp2',S.cam_pitch, 0.3,5,85,'P%.0f')
        if cv: S.cam_pitch=vv
        imgui.pop_item_width(); imgui.same_line(spacing=2)
        imgui.push_item_width(62)
        cv,vv=imgui.drag_float('##cd2',S.cam_dist,  0.05,0.3,15,'D%.1f')
        if cv: S.cam_dist=vv
        imgui.pop_item_width(); imgui.same_line(spacing=4)
        if imgui.button('Rst##rc3'): S.cam_yaw,S.cam_pitch,S.cam_dist=45,25,3.5
        imgui.text_colored('OVR', 0.5,0.6,0.75); imgui.same_line(spacing=6)
        _,S.show_forces      =imgui.checkbox('Frc##s', S.show_forces);      imgui.same_line(spacing=6)
        _,S.show_contacts    =imgui.checkbox('Ctt##s', S.show_contacts);    imgui.same_line(spacing=6)
        _,S.show_limits      =imgui.checkbox('Lim##s', S.show_limits);      imgui.same_line(spacing=6)
        _,S.show_grid        =imgui.checkbox('Grd##s', S.show_grid);        imgui.same_line(spacing=6)
        _,S.show_trajectory  =imgui.checkbox('Trj##s', S.show_trajectory);  imgui.same_line(spacing=6)
        _,S.show_joint_frames=imgui.checkbox('JFr##s', S.show_joint_frames)
        imgui.text_colored('FOG', 0.5,0.6,0.75); imgui.same_line(spacing=6)
        imgui.push_item_width(88)
        cv,vv=imgui.drag_float('##fg1s2',S.fog_start,0.2,0.5,50,'Strt %.1f')
        if cv: S.fog_start=vv
        imgui.pop_item_width(); imgui.same_line(spacing=4)
        imgui.push_item_width(88)
        cv,vv=imgui.drag_float('##fg2s2',S.fog_end,0.4,1,200,'End %.1f')
        if cv: S.fog_end=max(S.fog_start+1,vv)
        imgui.pop_item_width()


        imgui.end()

        # ══════════════════════════════════════════════════════════════════════
        # BOTTOM: Control Bar
        # ══════════════════════════════════════════════════════════════════════
        imgui.set_next_window_position(0, win.height - bot_h)
        imgui.set_next_window_size(win.width, bot_h)
        imgui.begin('##controls', flags | imgui.WINDOW_NO_SCROLLBAR)

        n_fr = S.n_frames

        # ── Row 1: playback controls + time info ────────────────────────────
        if imgui.button('|<##a',28,0): S.frame=S.loop_start; S._sync_cam(); G.sync_to_main()
        imgui.same_line(spacing=2)
        if imgui.button('<<##b',28,0): S.step(-10)
        imgui.same_line(spacing=2)
        if imgui.button('< ##c',28,0): S.step(-1)
        imgui.same_line(spacing=2)
        # Mode toggle: MAIN plays S loop, GRAPH plays G's ±50 loop
        mode_lbl = 'GRAPH' if G.play_mode == 'graph' else 'MAIN '
        try:
            mc = (0.2,0.72,0.25,1) if G.play_mode=='graph' else (0.22,0.42,0.82,1)
            imgui.push_style_color(getattr(imgui,'COLOR_BUTTON',21), *mc)
            if imgui.button(f'{mode_lbl}##pm', 46, 0):
                G.set_mode('graph' if G.play_mode=='main' else 'main')
            imgui.pop_style_color()
        except:
            if imgui.button(f'{mode_lbl}##pm', 46, 0):
                G.set_mode('graph' if G.play_mode=='main' else 'main')
        imgui.same_line(spacing=2)
        if G.play_mode == 'main':
            play_lbl = 'Pause' if S.playing else 'Play '
            if imgui.button(f'{play_lbl}##d',46,0): S.toggle_play()
        else:
            play_lbl = 'Pause' if G._playing else 'Play '
            if imgui.button(f'{play_lbl}##d',46,0): G.toggle_graph_play(S.n_frames)
        imgui.same_line(spacing=2)
        if imgui.button('> ##e',28,0): S.step(1)
        imgui.same_line(spacing=2)
        if imgui.button('>>##f',28,0): S.step(10)
        imgui.same_line(spacing=2)
        if imgui.button('>|##g',28,0): S.frame=S._hi(); S._sync_cam(); G.sync_to_main()
        imgui.same_line(spacing=8)
        imgui.push_item_width(120)
        ch,v=imgui.drag_int('##fr',S.frame,1.0,0,n_fr-1,'Frame %d')
        if ch: S.frame=max(0,min(n_fr-1,v)); S._sync_cam(); G.sync_to_main()
        imgui.pop_item_width()
        imgui.same_line(spacing=4)
        imgui.push_item_width(90)
        ch,v=imgui.drag_float('##sp',S.play_speed,0.02,0.1,5.0,'%.2fx')
        if ch: S.play_speed=max(0.05,v)
        imgui.pop_item_width()
        imgui.same_line(spacing=8)
        if S.data is not None:
            imgui.text_disabled(
                f't={float(S.data["time"][S.frame]):.2f}s'
                f'  vx={float(S.data["desired_vel_x"][S.frame]):.2f}')

        # ── Row 2: timeline bar (full width) ───────────────────────────────
        imgui.spacing()
        lo2=S.loop_start; hi2=S._hi()
        dl2=imgui.get_window_draw_list()
        bx,by=imgui.get_cursor_screen_pos()
        # Leave room for in/out/reset buttons on the right
        btn_area = 150
        bw2 = imgui.get_content_region_available_width() - btn_area
        bh2 = max(30, bot_h - 80)
        nlo2,nhi2,nfr2,ch2=_timeline_bar(dl2,bx,by,bw2,bh2,n_fr,lo2,hi2,S.frame)
        if ch2:
            S.loop_start=nlo2; S.loop_end=nhi2
            S.frame=max(0,min(n_fr-1,nfr2)); S._sync_cam(); G.sync_to_main()
        imgui.invisible_button('##tl2',bw2,bh2)
        imgui.set_cursor_screen_pos((bx+bw2+6, by))
        if imgui.button('[In]##si2', 44,0):  S.loop_start=S.frame
        imgui.same_line(spacing=2)
        if imgui.button('[Out]##so2',44,0): S.loop_end=S.frame
        if imgui.button('Reset##lr2', 44,0): S.loop_start=0; S.loop_end=0
        imgui.same_line(spacing=4)
        imgui.text_disabled(f'{lo2}–{hi2}')

        imgui.end()

        # ── Splitter overlays ─────────────────────────────────────────────────
        sv_x = vp_w; sh_y = win.height - bot_h
        bg = imgui.get_background_draw_list()
        bg.add_rect_filled(sv_x,0, sv_x+_SPLITTER_W, sh_y,
            imgui.get_color_u32_rgba(*(0.45,0.65,1.0,0.8) if _layout['drag_v'] else (0.22,0.22,0.26,0.7)))
        bg.add_rect_filled(0,sh_y, win.width, sh_y+_SPLITTER_W,
            imgui.get_color_u32_rgba(*(0.45,0.65,1.0,0.8) if _layout['drag_h'] else (0.22,0.22,0.26,0.7)))

        imgui.render()
        try: _imgui_renderer.render(imgui.get_draw_data())
        except Exception: pass

    # ── Events ────────────────────────────────────────────────────────────────

    def _do_reload(dt):
        """Close viewer, show picker, reload data into same AppState, reopen viewer."""
        # This runs in a scheduled callback, safe to close the window
        win.close()

    # After win.close() pyglet.app.run() returns, main loop handles restart.
    # We signal restart by setting a module-level flag read by run_pyglet's caller.


    _drag_in_vp = False

    def _in_vp(x):
        return x < _vp_w()

    @win.event
    def on_resize(width, height):
        _io.display_size=(width,height)

    @win.event
    def on_mouse_motion(x,y,dx,dy):
        nonlocal _mx,_my
        _mx=x; _my=y
        # Show resize cursor near splitters
        try:
            sv_x  = _vp_w()
            bot_h = max(80, int(win.height * _layout['bot_frac']))
            py_gl = win.height - y
            tl_top = win.height - bot_h
            if abs(x - sv_x) < _SPLITTER_W + 4 and py_gl < tl_top:
                win.set_mouse_cursor(win.get_system_mouse_cursor(win.CURSOR_SIZE_LEFT_RIGHT))
            elif abs(py_gl - tl_top) < _SPLITTER_W + 4:
                win.set_mouse_cursor(win.get_system_mouse_cursor(win.CURSOR_SIZE_UP_DOWN))
            else:
                win.set_mouse_cursor(None)
        except Exception: pass
        _in_bot = (win.height - y) < max(80, int(win.height * _layout['bot_frac']))
        if _in_bot:
            S.mouse_region = Region.PANEL
        elif _in_vp(x):
            S.mouse_region = Region.VIEWPORT
        else:
            S.mouse_region = Region.PANEL

    @win.event
    def on_mouse_press(x,y,btn,mods):
        nonlocal _drag_in_vp,_mx,_my
        _mx=x; _my=y
        from pyglet.window import mouse as pm
        if btn==pm.LEFT:
            _mb[0]=True
            # Check if clicking on a splitter
            sv_x = _vp_w()
            bot_h = max(80, int(win.height * _layout['bot_frac']))
            py_gl = win.height - y
            tl_top = win.height - bot_h
            if abs(x - sv_x) < _SPLITTER_W + 4 and py_gl < tl_top:
                _layout['drag_v'] = True
            elif abs(py_gl - tl_top) < _SPLITTER_W + 4:
                _layout['drag_h'] = True
        elif btn==pm.RIGHT: _mb[2]=True
        elif btn==pm.MIDDLE:_mb[1]=True
        try: _io.mouse_pos=(_mx,win.height-_my)
        except Exception: pass
        try:
            _io.mouse_down[0]=_mb[0]
            _io.mouse_down[1]=_mb[1]
            _io.mouse_down[2]=_mb[2]
        except Exception: pass
        # Only start a camera drag if click originated in viewport AND imgui doesn't want it
        # Only start camera drag if in viewport, above bottom bar, not on splitter
        bot_h2 = max(80, int(win.height * _layout['bot_frac']))
        in_bot = (win.height - y) < bot_h2
        on_splitter = (_layout['drag_v'] or _layout['drag_h'])
        _drag_in_vp = (S.mouse_region==Region.VIEWPORT
                       and not in_bot
                       and not on_splitter
                       and not _io.want_capture_mouse)

    @win.event
    def on_mouse_release(x,y,btn,mods):
        nonlocal _drag_in_vp,_mx,_my
        _mx=x; _my=y
        from pyglet.window import mouse as pm
        if btn==pm.LEFT:
            _layout['drag_v'] = False
            _layout['drag_h'] = False
        from pyglet.window import mouse as pm
        if btn==pm.LEFT:    _mb[0]=False
        elif btn==pm.RIGHT: _mb[2]=False
        elif btn==pm.MIDDLE:_mb[1]=False
        try:
            _io.mouse_down[0]=_mb[0]
            _io.mouse_down[1]=_mb[1]
            _io.mouse_down[2]=_mb[2]
        except Exception: pass
        _drag_in_vp=False

    @win.event
    def on_mouse_drag(x,y,dx,dy,btn,mods):
        nonlocal _mx,_my
        _mx=x; _my=y
        from pyglet.window import mouse as pm
        # Handle splitter resizing first
        if _layout['drag_v'] and btn==pm.LEFT:
            _layout['vp_frac'] = max(0.2, min(0.75, x / max(win.width,1)))
            return
        if _layout['drag_h'] and btn==pm.LEFT:
            py_gl = win.height - y  # top-down y
            _layout['bot_frac'] = max(0.06, min(0.35, (win.height - py_gl) / max(win.height,1)))
            return
        if not _drag_in_vp: return
        from pyglet.window import mouse as pm
        if btn==pm.LEFT:
            S.cam_yaw  -=dx*0.3
            S.cam_pitch =max(5,min(85,S.cam_pitch-dy*0.3))
        elif btn==pm.RIGHT:
            yr=math.radians(S.cam_yaw)
            r=np.array([-math.sin(yr),math.cos(yr),0.0])
            S.cam_target+=r*(-dx*0.004)+np.array([0,0,1])*(dy*0.004)

    @win.event
    def on_mouse_scroll(x,y,sx,sy):
        try: _io.mouse_wheel=float(sy)
        except Exception: pass
        if S.mouse_region==Region.PANEL: return
        S.cam_dist=max(0.3,S.cam_dist-sy*0.3)

    @win.event
    def on_text(text):
        try:
            for c in text: _io.add_input_character(ord(c))
        except Exception: pass

    @win.event
    def on_key_press(sym,mods):
        if _io.want_capture_keyboard: return
        k=pyglet.window.key
        if sym==k.SPACE:
            if G.play_mode=='graph': G.toggle_graph_play(S.n_frames)
            else: S.toggle_play()
        elif sym==k.LEFT:
            if G.play_mode == 'graph': G.step(-1, S.n_frames)
            else: S.step(-1)
        elif sym==k.RIGHT:
            if G.play_mode == 'graph': G.step(1, S.n_frames)
            else: S.step(1)
        elif sym==k.ESCAPE:win.close()
        elif sym==k.F6:    # MuJoCo: cycle frame visualization
            S.frame_cycle = (S.frame_cycle + 1) % 3
            S.show_joint_frames = S.frame_cycle > 0
        elif sym==k.F7:    # MuJoCo: cycle label (we use it to toggle limit warnings)
            S.show_limits = not S.show_limits

    @win.event
    def on_draw():
        nonlocal _mx,_my
        _in_bot_bar = (win.height - _my) < max(80, int(win.height * _layout['bot_frac']))
        S.mouse_region = (Region.PANEL if (_in_bot_bar or not _in_vp(_mx))
                          else Region.VIEWPORT)
        render_3d()
        glViewport(0,0,win.width,win.height)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        render_panel()

    def _update(dt):
        S.advance()
        if S.data is not None:
            dt_sim = float(S.data['time'][1] - S.data['time'][0])
            G.advance_graph(S.n_frames, S.play_speed, dt_sim)
    pyglet.clock.schedule_interval(_update, 1/120.0)
    print("Pyglet running — ESC to quit")
    pyglet.app.run()
    try: _imgui_renderer.shutdown()
    except Exception: pass


# =============================================================================
#  MAIN
# =============================================================================

# =============================================================================
#  ROBOT / DATA PICKER  (pyglet + imgui startup window)
# =============================================================================

def find_robots_dir():
    """Search for a 'robots' directory starting from cwd, then parent dirs."""
    p = Path.cwd()
    for _ in range(4):
        candidate = p / 'robots'
        if candidate.is_dir():
            return candidate
        p = p.parent
    return Path('robots')  # fallback, may not exist


def scan_robots(robots_dir: Path):
    """
    Return list of dicts:  {name, robot_dir, urdf_path, data_dirs}
    Expects structure:  robots/<name>/  with subdirs  go*/urdf/*.urdf  and  data/
    """
    robots = []
    if not robots_dir.is_dir():
        return robots
    for robot_bundle in sorted(robots_dir.iterdir()):
        if not robot_bundle.is_dir():
            continue
        # Find urdf file anywhere inside the bundle (not inside data/)
        urdf_path = None
        for p in sorted(robot_bundle.rglob('*.urdf')):
            if 'data' not in p.parts:
                urdf_path = p
                break
        if urdf_path is None:
            continue
        # Find data subdirectories (named 'data' or contain time.txt)
        data_dirs = []
        # Direct data/ subfolder
        direct_data = robot_bundle / 'data'
        if direct_data.is_dir():
            # Check for named sub-runs inside data/
            sub_runs = [d for d in sorted(direct_data.iterdir())
                        if d.is_dir() and (d / 'time.txt').exists()]
            if sub_runs:
                data_dirs.extend(sub_runs)
            elif (direct_data / 'time.txt').exists():
                data_dirs.append(direct_data)
        # Also look for any other dir containing time.txt
        for d in sorted(robot_bundle.iterdir()):
            if d.is_dir() and d.name != 'data' and (d / 'time.txt').exists():
                data_dirs.append(d)
        robots.append({
            'name':      robot_bundle.name,
            'bundle':    robot_bundle,
            'urdf':      urdf_path,
            'data_dirs': data_dirs,
        })
    return robots


def run_picker():
    """
    Show a small imgui window to pick robot + data folder, then return
    (urdf_path_str, data_dir_str) or raise SystemExit if cancelled.
    """
    try:
        import pyglet
        import imgui
        try:
            from imgui.integrations.opengl import ProgrammablePipelineRenderer as _PickRend
        except ImportError:
            from imgui_bundle.integrations.opengl import ProgrammablePipelineRenderer as _PickRend
        from pyglet.gl import (glClearColor, glClear, glViewport, glDisable,
                               GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST)
    except ImportError as e:
        print(f"Cannot show picker: {e}")
        return None, None

    robots_dir = find_robots_dir()
    robots     = scan_robots(robots_dir)

    W, H = 560, 380
    cfg  = pyglet.gl.Config(double_buffer=True, depth_size=24,
                            major_version=3, minor_version=3,
                            forward_compatible=True)
    win  = pyglet.window.Window(width=W, height=H,
                                caption='Go2 Viewer — Select Robot & Data',
                                resizable=False, config=cfg)

    imgui.create_context()
    io = imgui.get_io()
    try:    io.ini_file_name = b''
    except: pass
    io.display_size = (W, H)
    rend = _PickRend()

    state = {
        'robot_idx': 0,
        'data_idx':  0,
        'result':    None,   # (urdf, data) when Load pressed
        'cancelled': False,
        'mx': W//2, 'my': H//2,
        'mb': [False, False, False],
    }

    flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
             imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE)

    @win.event
    def on_draw():
        glViewport(0, 0, W, H)
        glClearColor(0.10, 0.10, 0.13, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)

        io.display_size = (W, H)
        try:    io.mouse_pos   = (state['mx'], H - state['my'])
        except: pass
        try:
            io.mouse_down[0] = state['mb'][0]
            io.mouse_down[1] = state['mb'][1]
            io.mouse_down[2] = state['mb'][2]
        except: pass
        imgui.new_frame()

        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(W, H)
        imgui.begin('##picker', flags=flags)

        imgui.text_colored('Go2 Viewer', 0.55, 0.75, 1.0)
        imgui.same_line(spacing=10)
        imgui.text_disabled('Select robot and data, then click Load')
        imgui.separator()
        imgui.spacing()

        if not robots:
            imgui.text_colored(
                f'No robots found in:  {robots_dir}', 1.0, 0.4, 0.4)
            imgui.spacing()
            imgui.text_disabled('Expected layout:')
            imgui.text_disabled('  robots/')
            imgui.text_disabled('    <robot_name>/')
            imgui.text_disabled('      go2/urdf/go2.urdf   (any *.urdf)')
            imgui.text_disabled('      data/               (contains time.txt)')
        else:
            # Robot picker
            imgui.text_colored('ROBOT', 0.65, 0.75, 0.88)
            robot_names = [r['name'] for r in robots]
            imgui.push_item_width(W - 30)
            ch, state['robot_idx'] = imgui.listbox(
                '##rb', state['robot_idx'], robot_names, min(6, len(robot_names)))
            imgui.pop_item_width()
            if ch:
                state['data_idx'] = 0  # reset data selection on robot change

            imgui.spacing()
            ri = state['robot_idx']
            robot = robots[ri]
            imgui.text_colored('URDF', 0.5, 0.6, 0.5)
            imgui.same_line(spacing=6)
            imgui.text_disabled(str(robot['urdf'].relative_to(robots_dir.parent)
                                    if robots_dir.parent in robot['urdf'].parents
                                    else robot['urdf']))

            imgui.spacing()
            # Data folder picker — only shows data for selected robot
            imgui.text_colored('DATA', 0.65, 0.75, 0.88)
            data_dirs = robot['data_dirs']
            if not data_dirs:
                imgui.text_colored('  No data/ folders found inside this robot bundle.',
                                   1.0, 0.5, 0.2)
            else:
                data_names = [str(d.relative_to(robot['bundle']))
                              for d in data_dirs]
                imgui.push_item_width(W - 30)
                _, state['data_idx'] = imgui.listbox(
                    '##dd', state['data_idx'],
                    data_names, min(5, len(data_names)))
                imgui.pop_item_width()

            imgui.spacing(); imgui.separator(); imgui.spacing()

            # Load button (only active if data is available)
            can_load = bool(data_dirs)
            if not can_load:
                imgui.push_style_color(getattr(imgui,'COLOR_BUTTON',21),
                                       0.3, 0.3, 0.3, 1.0)
            if imgui.button('  Load  ##ld', 100, 32) and can_load:
                state['result'] = (
                    str(robot['urdf']),
                    str(data_dirs[state['data_idx']]))
                # Defer close — never call win.close() from inside on_draw
                pyglet.clock.schedule_once(lambda dt: win.close(), 0.05)
            if not can_load:
                try: imgui.pop_style_color()
                except: pass
            imgui.same_line(spacing=10)
            if imgui.button('  Quit  ##qt', 100, 32):
                state['cancelled'] = True
                pyglet.clock.schedule_once(lambda dt: win.close(), 0.05)

        imgui.end()
        imgui.render()
        try:    rend.render(imgui.get_draw_data())
        except: pass

    @win.event
    def on_mouse_motion(x, y, dx, dy):
        state['mx'] = x; state['my'] = y

    @win.event
    def on_mouse_press(x, y, btn, mods):
        state['mx'] = x; state['my'] = y
        from pyglet.window import mouse as pm
        if btn == pm.LEFT:   state['mb'][0] = True
        elif btn == pm.RIGHT: state['mb'][2] = True
        try:
            io.mouse_pos   = (x, H - y)
            io.mouse_down[0] = state['mb'][0]
        except: pass

    @win.event
    def on_mouse_release(x, y, btn, mods):
        from pyglet.window import mouse as pm
        if btn == pm.LEFT:   state['mb'][0] = False
        elif btn == pm.RIGHT: state['mb'][2] = False
        try: io.mouse_down[0] = state['mb'][0]
        except: pass

    @win.event
    def on_key_press(sym, mods):
        if sym == pyglet.window.key.ESCAPE:
            state['cancelled'] = True
            pyglet.clock.schedule_once(lambda dt: win.close(), 0.05)

    pyglet.app.run()
    try: rend.shutdown()
    except: pass

    if state['cancelled'] or state['result'] is None:
        print("No selection made. Exiting.")
        sys.exit(0)

    return state['result']


def do_load(urdf_path_str, data_dir_str):
    """Parse URDF, load data, set up AppState. Returns (joint_order, ...) tuple."""
    if not os.path.isfile(urdf_path_str):
        print(f"ERROR: URDF not found: {urdf_path_str}"); sys.exit(1)
    if not os.path.isdir(data_dir_str):
        print(f"ERROR: data dir not found: {data_dir_str}"); sys.exit(1)

    joint_order, joint_limits, joint_axes, joint_origins, link_parents =         parse_urdf(urdf_path_str)
    fk_fn, urdf_robot = build_fk_fn(urdf_path_str, joint_order)

    data, n = load_data(data_dir_str)
    if data['q'].shape[1] != len(joint_order):
        m = min(data['q'].shape[1], len(joint_order))
        print(f"WARNING: joint count mismatch ({data['q'].shape[1]} vs {len(joint_order)}), using {m}")
        joint_order = joint_order[:m]

    S.data, S.n_frames = data, n
    S.cam_target = np.array([float(data['torso_x'][0]),
                              float(data['torso_y'][0]),
                              float(data['torso_z'][0])])
    S.cam_dist  = 2.5
    S.cam_pitch = 22.0
    G.sync_to_main()
    return urdf_path_str, joint_order, joint_limits, joint_axes, joint_origins, fk_fn, urdf_robot


def main():
    parser = argparse.ArgumentParser(
        description='Go2 Robot Viewer — omit --urdf/--data to use the interactive picker')
    parser.add_argument('--data', default=None, help='Path to data directory (optional)')
    parser.add_argument('--urdf', default=None, help='Path to URDF file (optional)')
    args = parser.parse_args()

    if args.urdf and args.data:
        # CLI mode — skip picker
        urdf_str  = args.urdf
        data_str  = args.data
    else:
        # Interactive picker
        print("Opening robot/data picker...")
        urdf_str, data_str = run_picker()

    while True:
        urdf_str, joint_order, joint_limits, joint_axes, joint_origins, fk_fn, urdf_robot = \
            do_load(urdf_str, data_str)

        wants_reload = run_pyglet(
            urdf_str, joint_order, joint_limits, joint_axes,
            joint_origins, fk_fn, urdf_robot)

        if not wants_reload:
            break
        # Show picker again for different robot/data
        print("Reload requested — showing picker...")
        S.data = None; S.n_frames = 0; S.frame = 0; S.playing = False
        urdf_str, data_str = run_picker()


if __name__ == '__main__':
    main()

