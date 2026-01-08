"""Evaluate a trained policy."""
from absl import app
from absl import flags

from datetime import datetime
import os
import pickle
import time

from isaacgym.torch_utils import to_torch  # pylint: disable=unused-import
from rsl_rl.runners import OnPolicyRunner
import torch
import yaml

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def calculate_cot_per_stride2(joint_torques, joint_velocities, dt, x_vel_body_avg, stride_duration, robot_mass, g=9.81):
    """
    Calculate Cost of Transport and velocity for a single stride.
    Args:
        joint_torques: Array of shape (num_joints, num_timesteps) - torques during stride
        joint_velocities: Array of shape (num_joints, num_timesteps) - velocities during stride
        dt: Time step between samples (seconds)
        x_vel_body_avg: Average forward velocity in body frame during stride (m/s)
        stride_duration: Duration of the stride (seconds)
        robot_mass: Total mass of the robot (kg)
        g: Gravitational acceleration (m/s², default 9.81)
    Returns:
        velocity: Average forward velocity during stride (m/s)
        cot: Cost of Transport (dimensionless)
    """
    # Use the average body-frame forward velocity
    velocity = x_vel_body_avg
    distance = velocity * stride_duration

    # Calculate mechanical work
    joint_work = np.sum(joint_torques * joint_velocities * dt, axis=1)
    mechanical_work = np.sum(np.abs(joint_work))

    # Calculate gravitational work
    gravitational_work = robot_mass * g * np.abs(distance)

    # COT = mechanical work / gravitational work
    cot = mechanical_work / gravitational_work

    return velocity, cot

def calculate_cot_per_stride(joint_torques, joint_velocities, dt, x_start, x_end, time_start, time_end, robot_mass, g=9.81):
  """
    Calculate Cost of Transport and velocity for a single stride.
    Args:
        joint_torques: Array of shape (num_joints, num_timesteps) - torques during stride
        joint_velocities: Array of shape (num_joints, num_timesteps) - velocities during stride
        dt: Time step between samples (seconds)
        x_start: Torso X position at stride start (meters)
        x_end: Torso X position at stride end (meters)
        time_start: Time at stride start (seconds)
        time_end: Time at stride end (seconds)
        robot_mass: Total mass of the robot (kg)
        g: Gravitational acceleration (m/s², default 9.81)
    Returns:
        velocity: Average velocity during stride (m/s)
        cot: Cost of Transport (dimensionless)
    """
  # Calculate stride velocity
  distance = x_end - x_start
  time_diff = time_end - time_start
  velocity = distance / time_diff

  # Calculate mechanical work: sum of |torque * velocity * dt| over all joints and timesteps
  # mechanical_work = np.sum(np.abs(joint_torques * joint_velocities * dt))

  joint_work = np.sum(joint_torques * joint_velocities * dt, axis=1)  # Sum over time for each joint
  mechanical_work = np.sum(np.abs(joint_work))  # Then take absolute value and sum joints


  # Calculate gravitational work: mass * g * distance
  gravitational_work = robot_mass * g * np.abs(distance)

  # COT = mechanical work / gravitational work
  cot = mechanical_work / gravitational_work

  return velocity, cot

# Add these new methods and modifications to the RealtimePlotter class

class RealtimePlotter:
    """Lightweight real-time plotter for simulation data with minimal overhead."""

    def __init__(self, max_points=1000, update_interval=0.05, initial_offset=None, swing_ratio=None):
        """
        Args:
            max_points: Maximum number of points to display
            update_interval: Update plot every N seconds (reduces overhead)
            initial_offset: List of 4 phase offsets for each leg [RL, FL, FR, RR]
            swing_ratio: List of 4 swing ratios for each leg [RL, FL, FR, RR]
        """
        self.max_points = max_points
        self.update_interval = update_interval
        self.last_update_time = 0
        
        # Store gait parameters
        self.initial_offset = initial_offset
        self.swing_ratio = swing_ratio

        # Create figure with 2x2 subplots
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.tight_layout(pad=3.0)

        # Initialize empty data containers
        self.desired_vel_data = {'x': deque(maxlen=max_points), 'y': deque(maxlen=max_points)}
        self.velocity_data = {'x': deque(maxlen=max_points), 'y': deque(maxlen=max_points)}
        self.stride_data = {'x': deque(maxlen=max_points), 'y': deque(maxlen=max_points)}
        self.stride_markers = {'x': [], 'y': []}
        self.cot_velocity_data = {'x': deque(maxlen=max_points), 'y': deque(maxlen=max_points)}

        # Foot contact history for real-time visualization
        self.contact_history = {
            'time': deque(maxlen=max_points),
            'contacts': deque(maxlen=max_points)  # Each entry is [RL, FL, FR, RR]
        }

        # Setup plots
        self._setup_combined_plot()
        self._setup_cot_plot()
        self._setup_gait_pattern_plot()
        self._setup_realtime_contacts_plot()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _setup_gait_pattern_plot(self):
        """Setup gait pattern visualization showing expected stance/swing phases."""
        ax = self.axes[1, 0]  # Bottom-left
        ax.set_xlabel('Phase (normalized)')
        ax.set_ylabel('Leg')
        ax.set_title('Expected Gait Pattern (Blue=Stance)')
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 3.5)
        
        # Order: Rear Left (top, y=3), Front Left (y=2), Front Right (y=1), Rear Right (bottom, y=0)
        leg_names = ['Rear Right', 'Front Right', 'Front Left', 'Rear Left']  # y=0,1,2,3
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(leg_names)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Leg order in initial_offset/swing_ratio arrays: [RL, FL, FR, RR] -> indices 0,1,2,3
        # Map to y-positions: RL->y=3, FL->y=2, FR->y=1, RR->y=0
        leg_to_y = {0: 3, 1: 2, 2: 1, 3: 0}  # RL=0->y=3, FL=1->y=2, FR=2->y=1, RR=3->y=0

        bar_height = 0.6
        for leg_idx, (offset, swing) in enumerate(zip(self.initial_offset, self.swing_ratio)):
            y_pos = leg_to_y[leg_idx]
            stance_duration = 1 - swing
            stance_start = offset  # Stance begins at offset
            
            # Handle wrap-around (stance phase crosses phase=1.0 boundary)
            if stance_start + stance_duration <= 1.0:
                # No wrap - single bar
                ax.barh(y_pos, stance_duration, left=stance_start, height=bar_height,
                       color='#4BACC6', edgecolor='black', linewidth=0.5)
            else:
                # Wrap-around - need TWO bars
                # First part: from stance_start to 1.0
                first_part_width = 1.0 - stance_start
                ax.barh(y_pos, first_part_width, left=stance_start, height=bar_height,
                       color='#4BACC6', edgecolor='black', linewidth=0.5)
                # Second part: from 0 to the remainder
                second_part_width = stance_duration - first_part_width  # = (stance_start + stance_duration) - 1.0
                ax.barh(y_pos, second_part_width, left=0, height=bar_height,
                       color='#4BACC6', edgecolor='black', linewidth=0.5)

    def _setup_realtime_contacts_plot(self):
        """Setup real-time foot contact visualization."""
        ax = self.axes[1, 1]  # Bottom-right
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Leg')
        ax.set_title('Real-time Foot Contacts (Blue=Contact)')
        ax.set_ylim(-0.5, 3.5)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['Rear Left', 'Front Left', 'Front Right', 'Rear Right'])
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 5)  # Initial time window

        # Store scatter plot reference for updates
        self.contact_scatters = [ax.scatter([], [], c='#4BACC6', marker='s', s=100) for _ in range(4)]

    def add_contact_data(self, time_val, contacts):
        """
        Add foot contact data point.
        Args:
            time_val: Current simulation time
            contacts: List of 4 booleans [RL, FL, FR, RR] - True if in contact
        """
        self.contact_history['time'].append(time_val)
        self.contact_history['contacts'].append(contacts.copy())

    def _update_contacts_plot(self):
        """Update the real-time contacts visualization."""
        if len(self.contact_history['time']) == 0:
            return

        ax = self.axes[1, 1]
        times = list(self.contact_history['time'])
        contacts = list(self.contact_history['contacts'])

        # Update each leg's scatter plot
        for leg_idx in range(4):
            # Get times when this leg is in contact
            contact_times = [t for t, c in zip(times, contacts) if c[leg_idx]]
            contact_y = [leg_idx] * len(contact_times)

            if contact_times:
                self.contact_scatters[leg_idx].set_offsets(
                    np.c_[contact_times, contact_y]
                )

        # Auto-scale x-axis to show last 5 seconds
        if times:
            max_time = max(times)
            ax.set_xlim(max(0, max_time - 5), max_time + 0.5)

    def _update_plot(self):
        """Update the plots efficiently."""
        try:
            ax1 = self.axes[0, 0]  # Changed to use 2D indexing

            # Update velocity plot
            if len(self.velocity_data['x']) > 0:
                self.velocity_line.set_data(
                    list(self.velocity_data['x']),
                    list(self.velocity_data['y'])
                )
                self._auto_scale_axis(ax1, self.velocity_data)

            if len(self.desired_vel_data['x']) > 0:
                self.desired_velocity_line.set_data(
                    list(self.desired_vel_data['x']),
                    list(self.desired_vel_data['y'])
                )
                self._auto_scale_axis_secondary(self.ax3, self.desired_vel_data)

            if len(self.stride_data['x']) > 0:
                self.stride_line.set_data(
                    list(self.stride_data['x']),
                    list(self.stride_data['y'])
                )
                self._auto_scale_axis_secondary(self.ax2, self.stride_data)

            self._update_stride_markers()

            # Update COT scatter plot
            if len(self.cot_velocity_data['x']) > 0:
                self.cot_scatter.set_offsets(
                    np.c_[list(self.cot_velocity_data['x']), list(self.cot_velocity_data['y'])]
                )
                self._auto_scale_cot_plot(self.axes[0, 1], self.cot_velocity_data)

            # Update contacts plot
            self._update_contacts_plot()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception as e:
            print(f"Plot update error: {e}")

    def _setup_combined_plot(self):
        """Setup combined velocity and stride length plot with dual y-axes."""
        ax1 = self.axes[0, 0]  # Changed to 2D indexing
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Velocity (m/s)', color='b')
        ax1.set_title('Velocity and Stride Length vs Time')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='y', labelcolor='b')

        self.velocity_line, = ax1.plot([], [], 'b-', linewidth=2, label='Velocity')
        self.desired_velocity_line, = ax1.plot([], [], 'r-', linewidth=2, label='Desired Velocity')

        self.ax2 = ax1.twinx()
        self.ax2.set_ylabel('Stride Length (m)', color='g')
        self.ax2.tick_params(axis='y', labelcolor='g')

        self.ax3 = ax1.twinx()
        self.ax3.spines['right'].set_position(('outward', 60))
        self.ax3.set_ylabel('Desired Velocity (m/s)', color='r')
        self.ax3.tick_params(axis='y', labelcolor='r')

        self.stride_line, = self.ax2.plot([], [], 'g-', linewidth=2, label='Stride Length')

        self.stride_lines = []
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 3)
        self.ax2.set_ylim(0, 2)

    def _setup_cot_plot(self):
        """Setup COT vs Average Velocity scatter plot."""
        ax = self.axes[0, 1]  # Changed to 2D indexing
        ax.set_xlabel('Average Velocity (m/s)')
        ax.set_ylabel('Cost of Transport (COT)')
        ax.set_title('Cost of Transport vs Average Velocity')
        ax.grid(True, alpha=0.3)

        self.cot_scatter = ax.scatter([], [], c='purple', marker='o', s=50, alpha=0.6, label='COT')
        ax.legend(loc='upper right')
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 5)

    def _update_stride_markers(self):
        """Update vertical lines for stride events on combined plot."""
        if len(self.stride_markers['x']) > 0:
            for line in self.stride_lines:
                line.remove()
            self.stride_lines.clear()

            for stride_time in self.stride_markers['x']:
                line = self.axes[0, 0].axvline(x=stride_time, color='red', linestyle='--',
                                               linewidth=1.5, alpha=0.7)
                self.stride_lines.append(line)

    def _auto_scale_axis(self, ax, data):
        """Auto-scale primary axis based on data."""
        if len(data['x']) > 0:
            x_data = list(data['x'])
            y_data = list(data['y'])
            max_time = max(x_data)
            ax.set_xlim(max(0, max_time - 10), max_time + 1)
            if y_data:
                y_min, y_max = min(y_data), max(y_data)
                margin = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 0.1
                ax.set_ylim(y_min - margin, y_max + margin)

    def _auto_scale_axis_secondary(self, ax, data):
        """Auto-scale secondary axis based on data."""
        if len(data['x']) > 0:
            y_data = list(data['y'])
            if y_data:
                y_min, y_max = min(y_data), max(y_data)
                margin = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 0.1
                ax.set_ylim(y_min - margin, y_max + margin)

    def _auto_scale_cot_plot(self, ax, data):
        """Auto-scale COT plot based on data."""
        if len(data['x']) > 0 and len(data['y']) > 0:
            x_data = list(data['x'])
            y_data = list(data['y'])
            x_min, x_max = min(x_data), max(x_data)
            x_margin = (x_max - x_min) * 0.1 if (x_max - x_min) > 0 else 0.1
            ax.set_xlim(max(0, x_min - x_margin), x_max + x_margin)
            y_min, y_max = min(y_data), max(y_data)
            y_margin = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 0.1
            ax.set_ylim(max(0, y_min - y_margin), y_max + y_margin)

class GaitStrideDetector:
    """
    Detects strides based on flight phases for different gallop types (G0, G2, GG, GE).
    Uses a sliding window approach to detect gait patterns.
    """
    
    def __init__(self, gait_type='G0', window_size=100):
        """
        Args:
            gait_type: One of 'G0', 'G2', 'GG', 'GE'
            window_size: Size of the sliding window buffer
        """
        self.gait_type = gait_type
        self.window_size = window_size
        
        # Sliding window buffer
        self.contact_buffer = deque(maxlen=window_size)
        self.time_buffer = deque(maxlen=window_size)
        
        # Track where we've already detected strides
        self.last_stride_end_idx = 0
        
    def add_contact(self, time_val, contacts):
        """
        Add a contact data point to the buffer.
        Args:
            time_val: Current simulation time
            contacts: List of 4 booleans [RL, FL, FR, RR]
        Returns:
            stride_info: Dict with stride info if detected, None otherwise
        """
        self.contact_buffer.append(contacts.copy())
        self.time_buffer.append(time_val)
        
        # Try to detect stride
        return self._detect_stride()
    
    def _count_flight_phases(self, contact_sequence):
        """
        Count the number of flight phases (all legs off ground) in a sequence.
        Returns: (num_flight_phases, flight_phase_indices)
        """
        flight_phases = []
        in_flight = False
        flight_start = None
        
        for i, contacts in enumerate(contact_sequence):
            all_off = not any(contacts)  # True if all legs are off ground
            
            if all_off and not in_flight:
                in_flight = True
                flight_start = i
            elif not all_off and in_flight:
                in_flight = False
                flight_phases.append((flight_start, i))
        
        # Handle case where sequence ends in flight
        if in_flight:
            flight_phases.append((flight_start, len(contact_sequence)))
            
        return len(flight_phases), flight_phases
    
    def _has_mixed_stance(self, contact_sequence):
        """
        Check if sequence has mixed stance phases (front and rear legs alternating).
        """
        for contacts in contact_sequence:
            # Mixed stance: front pair and rear pair both have at least one contact
            # but not all same
            front_contacts = contacts[1] or contacts[2]  # FL or FR
            rear_contacts = contacts[0] or contacts[3]   # RL or RR
            if front_contacts and rear_contacts:
                return True
        return False
    
    def _detect_stride(self):
        """
        Detect if a valid stride pattern exists in the buffer.
        Returns stride info dict if detected, None otherwise.
        """
        if len(self.contact_buffer) < 10:  # Need minimum samples
            return None
        
        # Convert buffer to list for analysis
        contacts = list(self.contact_buffer)
        times = list(self.time_buffer)
        
        # Look for stride pattern starting from last_stride_end_idx
        start_idx = max(0, self.last_stride_end_idx)
        
        if start_idx >= len(contacts) - 10:
            return None
        
        # Find a complete gait cycle
        # A stride is complete when we return to a similar contact state
        # after going through the expected flight phase pattern
        
        search_window = contacts[start_idx:]
        search_times = times[start_idx:]
        
        stride_info = self._find_stride_in_window(search_window, search_times, start_idx)
        
        if stride_info:
            self.last_stride_end_idx = stride_info['end_idx']
            return stride_info
        
        return None
    
    def _find_stride_in_window(self, contacts, times, global_offset):
        """
        Find a stride pattern in the given window based on gait type.
        """
        num_flights, flight_phases = self._count_flight_phases(contacts)
        
        # Determine expected pattern based on gait type
        if self.gait_type == 'G0':
            # G0: No flight phases, two mixed stance phases
            return self._detect_g0_stride(contacts, times, global_offset)
        elif self.gait_type == 'G2':
            # G2: Two flight phases
            return self._detect_g2_stride(contacts, times, global_offset, flight_phases)
        elif self.gait_type == 'GG':
            # GG: Single flight phase with gathered legs
            return self._detect_gg_stride(contacts, times, global_offset, flight_phases)
        elif self.gait_type == 'GE':
            # GE: Single flight phase with extended legs
            return self._detect_ge_stride(contacts, times, global_offset, flight_phases)
        else:
            # Default: simple contact pattern matching
            return self._detect_simple_stride(contacts, times, global_offset)
    
    def _detect_g0_stride(self, contacts, times, global_offset):
        """
        Detect G0 stride: No flight phases, characterized by alternating 
        front-rear stance phases.
        """
        # G0 has pattern: rear stance -> mixed -> front stance -> mixed -> rear stance
        # Look for transition through contact states without full flight
        
        # Find sequence with no flight phases but clear leg pair transitions
        state_sequence = []
        for i, c in enumerate(contacts):
            front_on = c[1] or c[2]
            rear_on = c[0] or c[3]
            all_off = not any(c)
            
            if all_off:
                return None  # G0 shouldn't have flight phases
            
            if front_on and rear_on:
                state = 'mixed'
            elif front_on:
                state = 'front'
            elif rear_on:
                state = 'rear'
            else:
                state = 'flight'
                
            if not state_sequence or state_sequence[-1][0] != state:
                state_sequence.append((state, i))
        
        # Look for pattern: need at least one complete cycle
        # rear -> mixed -> front -> mixed -> rear (or similar)
        if len(state_sequence) >= 4:
            # Find a complete cycle
            for i in range(len(state_sequence) - 3):
                states = [s[0] for s in state_sequence[i:i+4]]
                # Accept various valid G0 patterns
                if 'flight' not in states and len(set(states)) >= 2:
                    start_idx = state_sequence[i][1]
                    end_idx = state_sequence[i+3][1] if i+3 < len(state_sequence) else len(contacts) - 1
                    
                    if end_idx - start_idx >= 5:  # Minimum stride length
                        return {
                            'start_idx': global_offset + start_idx,
                            'end_idx': global_offset + end_idx,
                            'start_time': times[start_idx],
                            'end_time': times[end_idx],
                            'gait_type': 'G0',
                            'num_flight_phases': 0,
                            'valid': True
                        }
        
        return None
    
    def _detect_g2_stride(self, contacts, times, global_offset, flight_phases):
        """
        Detect G2 stride: Two flight phases per stride.
        """
        if len(flight_phases) < 2:
            return None
        
        # G2 has two flight phases - find a complete cycle
        # Look for pattern: stance -> flight -> stance -> flight -> stance
        for i in range(len(flight_phases) - 1):
            fp1_start, fp1_end = flight_phases[i]
            fp2_start, fp2_end = flight_phases[i + 1]
            
            # Verify there's stance between the two flight phases
            if fp2_start > fp1_end:
                # Check for stance phase between flights
                stance_between = False
                for j in range(fp1_end, fp2_start):
                    if any(contacts[j]):
                        stance_between = True
                        break
                
                if stance_between:
                    # Found valid G2 stride
                    # Stride starts before first flight, ends after second flight
                    start_idx = max(0, fp1_start - 5)
                    end_idx = min(len(contacts) - 1, fp2_end + 5)
                    
                    return {
                        'start_idx': global_offset + start_idx,
                        'end_idx': global_offset + end_idx,
                        'start_time': times[start_idx],
                        'end_time': times[end_idx],
                        'gait_type': 'G2',
                        'num_flight_phases': 2,
                        'valid': True
                    }
        
        return None
    
    def _detect_gg_stride(self, contacts, times, global_offset, flight_phases):
        """
        Detect GG stride: Single flight phase with gathered suspension.
        """
        if len(flight_phases) < 1:
            return None
        
        # GG has one flight phase - legs gathered
        # Look for: stance (rear leading) -> flight -> stance (front leading)
        for fp_start, fp_end in flight_phases:
            if fp_start < 5 or fp_end >= len(contacts) - 5:
                continue
            
            # Check stance before flight (should have rear legs)
            pre_flight_rear = any(contacts[fp_start-1][0] or contacts[fp_start-1][3] 
                                  for _ in range(1))
            
            # Check stance after flight (should have front legs)
            post_flight_front = any(contacts[min(fp_end, len(contacts)-1)][1] or 
                                   contacts[min(fp_end, len(contacts)-1)][2] 
                                   for _ in range(1))
            
            # For GG, accept if we have clear stance-flight-stance pattern
            start_idx = max(0, fp_start - 10)
            end_idx = min(len(contacts) - 1, fp_end + 10)
            
            # Verify no second flight phase in this window
            window_contacts = contacts[start_idx:end_idx]
            num_flights_in_window, _ = self._count_flight_phases(window_contacts)
            
            if num_flights_in_window == 1:
                return {
                    'start_idx': global_offset + start_idx,
                    'end_idx': global_offset + end_idx,
                    'start_time': times[start_idx],
                    'end_time': times[end_idx],
                    'gait_type': 'GG',
                    'num_flight_phases': 1,
                    'valid': True
                }
        
        return None
    
    def _detect_ge_stride(self, contacts, times, global_offset, flight_phases):
        """
        Detect GE stride: Single flight phase with extended suspension.
        """
        # GE is similar to GG but with different leg timing
        # For now, use similar detection logic
        return self._detect_gg_stride(contacts, times, global_offset, flight_phases)
    
    def _detect_simple_stride(self, contacts, times, global_offset):
        """
        Simple stride detection: look for specific contact pattern.
        Fallback method.
        """
        target_pattern = [False, False, True, True]  # Original pattern
        
        for i, c in enumerate(contacts):
            if c == target_pattern and i > 0:
                # Found target pattern
                return {
                    'start_idx': global_offset,
                    'end_idx': global_offset + i,
                    'start_time': times[0],
                    'end_time': times[i],
                    'gait_type': 'simple',
                    'num_flight_phases': -1,
                    'valid': True
                }
        
        return None
    
    def reset(self):
        """Reset the detector state."""
        self.contact_buffer.clear()
        self.time_buffer.clear()
        self.last_stride_end_idx = 0

from src.envs import env_wrappers
torch.set_printoptions(precision=2, sci_mode=False)

flags.DEFINE_string("logdir", None, "logdir.")
flags.DEFINE_bool("use_gpu", False, "whether to use GPU.")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_bool("use_real_robot", False, "whether to use real robot.")
flags.DEFINE_integer("num_envs", 1,
                     "number of environments to evaluate in parallel.")
flags.DEFINE_bool("save_traj", False, "whether to save trajectory.")
flags.DEFINE_bool("use_contact_sensor", True, "whether to use contact sensor.")
flags.DEFINE_bool("enable_plotting", True, "whether to enable real-time plotting.")
flags.DEFINE_float("plot_update_interval", 0.05, "plot update interval in seconds.")
FLAGS = flags.FLAGS


def get_latest_policy_path(logdir):
  files = [
    entry for entry in os.listdir(logdir)
    if os.path.isfile(os.path.join(logdir, entry))
  ]
  files.sort(key=lambda entry: os.path.getmtime(os.path.join(logdir, entry)))
  files = files[::-1]

  for entry in files:
    if entry.startswith("model"):
      return os.path.join(logdir, entry)
  raise ValueError("No Valid Policy Found.")


def main(argv):
  del argv  # unused

  device = "cuda" if FLAGS.use_gpu else "cpu"

  # Load config and policy
  if FLAGS.logdir.endswith("pt"):
    config_path = os.path.join(os.path.dirname(FLAGS.logdir), "config.yaml")
    policy_path = FLAGS.logdir
    root_path = os.path.dirname(FLAGS.logdir)
  else:
    # Find the latest policy ckpt
    config_path = os.path.join(FLAGS.logdir, "config.yaml")
    policy_path = get_latest_policy_path(FLAGS.logdir)
    root_path = FLAGS.logdir

  with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.Loader)

  with config.unlocked():
    velocity_up = torch.linspace(0.5, 4.0, 200) 
    # velocity_down = torch.linspace(4.0, 0.5, 50)
    # velocity_schedule = torch.cat([velocity_up, velocity_down], dim=0)
    velocity_schedule = velocity_up
    config.environment.jumping_distance_schedule = velocity_schedule / config.environment.gait.stepping_frequency
    # config.environment.jumping_distance_schedule = None
    config.environment.gait.desired_velocity = torch.tensor([velocity_schedule[0].item(), 0, 0])  # Start with first velocity


    # config.environment.jumping_distance_schedule = torch.linspace(0.3, 1.5, 100)
    config.environment.max_jumps = 3000

  env = config.env_class(num_envs=FLAGS.num_envs,
                         device=device,
                         config=config.environment,
                         show_gui=FLAGS.show_gui,
                         use_real_robot=FLAGS.use_real_robot)
  env = env_wrappers.RangeNormalize(env)
  if FLAGS.use_real_robot:
    env.robot.state_estimator.use_external_contact_estimator = (
      not FLAGS.use_contact_sensor)

  # Retrieve policy
  runner = OnPolicyRunner(env, config.training, policy_path, device=device)
  runner.load(policy_path)
  policy = runner.get_inference_policy()
  runner.alg.actor_critic.train()

  # Reset environment
  state, _ = env.reset()
  total_reward = torch.zeros(FLAGS.num_envs, device=device)
  steps_count = 0

  start_time = time.time()
  logs = []

  # Initialize plotter
  plotter = None
  max_strides = 10000
  stride_detector = None
  cot_data_arrays = {
    'velocity': np.zeros(max_strides, dtype=np.float32),
    'cot': np.zeros(max_strides, dtype=np.float32),
    # 'time': np.zeros(max_strides, dtype=np.float32),
    # 'stride_length': np.zeros(max_strides, dtype=np.float32)
  }
  cot_data_count = 0

  initial_offset = list(config.environment.gait.initial_offset)
  swing_ratio = list(config.environment.gait.swing_ratio)
  gait_name = config.environment.gait.gait_name
  gait_name = "G0_Transverse"
  # print(f"initial_offset : {initial_offset}")
  # print(f"swing_ratio : {swing_ratio}")
  # print(f"gait_name : {gait_name}")

  if FLAGS.enable_plotting:
    plotter = RealtimePlotter(
      max_points=10000, 
      update_interval=FLAGS.plot_update_interval,
      initial_offset=initial_offset,
      swing_ratio=swing_ratio
    )
    print(f"Real-time plotting enabled (update every {FLAGS.plot_update_interval}s)")
    print(f"Gait parameters - Initial offset: {initial_offset}, Swing ratio: {swing_ratio}")

  stride_detector = GaitStrideDetector(gait_type=gait_name, window_size=200)
  print(f"Stride detector initialized for gait type: {gait_name}")


  # Before the main loop - pre-allocate buffers
  max_stride_steps = 1000  # Adjust based on expected max stride duration
  num_joints = 12

  stride_torques = np.zeros((num_joints, max_stride_steps), dtype=np.float32)
  stride_velocities = np.zeros((num_joints, max_stride_steps), dtype=np.float32)
  stride_step_count = 0
  stride_x_start = 0.0
  stride_time_start = 0.0

  print("Starting simulation loop...")

  velocity_index = 0
  with torch.inference_mode():
    stride_x_velocities = []  #FIXED!: Track body-frame X velocities during stride
    while True:
      steps_count += 1
      action = policy(state)
      state, _, reward, done, info = env.step(action)

      # Extract current data
      current_time = env.robot.time_since_reset.item()
      velocity = torch.norm(env.robot.base_vel).item()
      current_contacts = env.robot.foot_contacts[0].tolist()
      # print(f"foot contacts: {current_contacts}")

      # Get body-frame velocity (forward velocity in robot's X-axis)
      body_frame_velocity = env.robot.base_velocity_body_frame[0]  # Shape: (3,)
      forward_velocity = body_frame_velocity[0].item()  # X-component in body frame


      # Collect data for COT calculation
      current_joint_torques = env.robot.motor_torques[0].cpu().numpy()  # Shape: (12,)
      current_joint_velocities = env.robot.motor_velocities[0].cpu().numpy()  # Shape: (12,)

      # Store in pre-allocated buffers (overwrite, don't append)
      if stride_step_count < max_stride_steps:
        stride_torques[:, stride_step_count] = current_joint_torques
        stride_velocities[:, stride_step_count] = current_joint_velocities
        stride_x_velocities.append(forward_velocity)  # Track forward velocity
        stride_step_count += 1

      # Extract stride length from environment
      stride_length = env._jumping_distance[0,0].item() if torch.is_tensor(env._jumping_distance[0]) else float(env._jumping_distance)
      # print(f"Current Stride Length: {stride_length:.3f}m, Time: {current_time:.2f}s")

      if plotter is not None:
        plotter.add_contact_data(current_time, current_contacts)

      # Use stride detector instead of simple pattern matching
      stride_info = stride_detector.add_contact(current_time, current_contacts)

      # Detect stride events
      is_stride_event = False
      if stride_info is not None and stride_info['valid']:
        is_stride_event = True
        print(f">>> STRIDE EVENT ({stride_info['gait_type']}) detected at time {current_time:.2f}s <<<")

        if velocity_index < len(velocity_schedule):
            env._desired_velocity[:, 0] = velocity_schedule[velocity_index]
            env._desired_velocity[:, 1:] = 0
            velocity_index += 1
        # print(f"DV: {env._desired_velocity[0]}")

        # FIXED!:Calculate COT2 for completed stride
        if stride_step_count > 0:
          avg_forward_velocity = np.mean(stride_x_velocities)  # This is correct - only current stride
          stride_duration = current_time - stride_time_start

          stride_velocity, stride_cot = calculate_cot_per_stride2(
            stride_torques[:, :stride_step_count],
            stride_velocities[:, :stride_step_count],
            env._config.env_dt,
            avg_forward_velocity,  # Average for this stride only
            stride_duration,
            env.robot.mass
          )

          print(f"Stride Velocity: {stride_velocity:.3f} m/s, Current Velocity: {forward_velocity:.3f} m/s, desired_velocity: {env._desired_velocity[0][0]}m/s, COT: {stride_cot:.4f}")

          # # Optional: Filter out invalid strides (like MATLAB does with COT < 20)
          # if stride_cot < 2.5:
          #   if cot_data_count < max_strides:
          #     cot_data_arrays['velocity'][cot_data_count] = stride_velocity
          #     cot_data_arrays['cot'][cot_data_count] = stride_cot
          #     # cot_data_arrays['time'][cot_data_count] = current_time
          #     # cot_data_arrays['stride_length'][cot_data_count] = stride_length
          #     cot_data_count += 1
          #
          #   if plotter is not None:
          #     plotter.add_cot_data(stride_velocity, stride_cot)

        # Reset for next stride (overwrite from beginning)
        stride_step_count = 0
        stride_x_start = env.robot.base_position[0, 0].item()
        stride_x_velocities = []  # CLEAR the list for the next stride
        stride_time_start = current_time

      # Update real-time plot (if enabled) - CALL EVERY STEP
      # print(velocity, type(velocity))
      # print(forward_velocity, type(forward_velocity))
      if plotter is not None:
        try:
          plotter.add_data(
            time_val=current_time,
            # desired_vel= velocity_schedule[velocity_index],
            desired_vel=velocity_schedule[velocity_index] if velocity_index < len(velocity_schedule) else velocity_schedule[-1],
            velocity=velocity, #  torch.norm(env.robot.base_vel).item()
            stride_length=stride_length,
            is_stride_event=is_stride_event
          )
        except:
            pass

  end_time = time.time()
  elapsed = end_time - start_time

  print(f"\n{'='*60}")
  print(f"Simulation Complete!")
  print(f"{'='*60}")
  print(f"Total reward: {total_reward.item():.2f}")
  print(f"Total steps: {steps_count}")
  print(f"Time elapsed: {elapsed:.2f}s")
  print(f"Steps per second: {steps_count / elapsed:.1f} Hz")
  print(f"{'='*60}\n")

  # Save COT data to CSV efficiently
  if cot_data_count > 0:
    # Trim arrays to actual data size
    cot_output_path = os.path.join(root_path, f"cot_data_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv")
    # Stack arrays and save (most efficient for numpy)
    data_to_save = np.column_stack([
      cot_data_arrays['velocity'][:cot_data_count],
      cot_data_arrays['cot'][:cot_data_count],
      # cot_data_arrays['time'][:cot_data_count],
      # cot_data_arrays['stride_length'][:cot_data_count]
    ])
    np.savetxt(cot_output_path, data_to_save, 
               delimiter=',', 
               header='velocity,cot',#,time,stride_length',
               comments='',
               fmt='%.6f')

    print(f"COT data saved to: {cot_output_path}")
    print(f"Total strides recorded: {cot_data_count}")

  if plotter is not None:
    print("Plot window is open. Press Enter to close and exit...")
    input()
    plotter.close()

  if FLAGS.use_real_robot or FLAGS.save_traj:
    mode = "real" if FLAGS.use_real_robot else "sim"
    output_dir = f"eval_{mode}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.pkl"
    output_path = os.path.join(root_path, output_dir)

    with open(output_path, "wb") as fh:
      pickle.dump(logs, fh)
    print(f"Data logged to: {output_path}")


if __name__ == "__main__":
  app.run(main)

