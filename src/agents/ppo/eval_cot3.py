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
    """
    velocity = x_vel_body_avg
    distance = velocity * stride_duration
    joint_work = np.sum(joint_torques * joint_velocities * dt, axis=1)
    mechanical_work = np.sum(np.abs(joint_work))
    gravitational_work = robot_mass * g * np.abs(distance)
    cot = mechanical_work / gravitational_work
    return velocity, cot


class GaitStrideDetector:
    """
    Detects strides based on flight phases for different gallop types.
    
    Contact array order: [FR, FL, RR, RL] (indices 0, 1, 2, 3)
    """
    
    def __init__(self, gait_type='g0_transverse', window_size=200, min_stride_samples=15):
        self.gait_type = gait_type.lower()
        self.window_size = window_size
        self.min_stride_samples = min_stride_samples
        
        parts = self.gait_type.split('_')
        self.gait_class = parts[0]
        self.gait_variant = parts[1] if len(parts) > 1 else 'transverse'
        
        self.expected_flight_phases = {
            'g0': 0,
            'gg': 1,
            'ge': 1,
            'g2': 2
        }.get(self.gait_class, 0)
        
        # Use lists instead of deques to avoid index issues
        self.contact_buffer = []
        self.time_buffer = []
        
        # Track position in buffer where we should start searching
        self.search_start_idx = 0
        self.stride_count = 0
        
        # Contact array order: [FR, FL, RR, RL]
        self.FRONT_RIGHT = 0
        self.FRONT_LEFT = 1
        self.REAR_RIGHT = 2
        self.REAR_LEFT = 3
    
    def add_contact(self, time_val, contacts):
        """
        Add a contact data point to the buffer and check for stride.
        """
        self.contact_buffer.append(list(contacts))
        self.time_buffer.append(time_val)
        
        # Keep buffer size manageable by trimming old data
        # But preserve search_start_idx validity
        max_buffer_size = self.window_size * 2
        if len(self.contact_buffer) > max_buffer_size:
            # Remove old entries and adjust search_start_idx
            trim_amount = len(self.contact_buffer) - self.window_size
            self.contact_buffer = self.contact_buffer[trim_amount:]
            self.time_buffer = self.time_buffer[trim_amount:]
            self.search_start_idx = max(0, self.search_start_idx - trim_amount)
        
        return self._detect_stride()
    
    def _is_flight_phase(self, contacts):
        """Check if all legs are off the ground (flight phase)."""
        return not any(contacts)
    
    def _is_hind_stance(self, contacts):
        """Check if only hind/rear legs are in contact."""
        rear_on = contacts[self.REAR_LEFT] or contacts[self.REAR_RIGHT]
        front_on = contacts[self.FRONT_LEFT] or contacts[self.FRONT_RIGHT]
        return rear_on and not front_on
    
    def _is_fore_stance(self, contacts):
        """Check if only front legs are in contact."""
        rear_on = contacts[self.REAR_LEFT] or contacts[self.REAR_RIGHT]
        front_on = contacts[self.FRONT_LEFT] or contacts[self.FRONT_RIGHT]
        return front_on and not rear_on
    
    def _is_mixed_stance(self, contacts):
        """Check if both front and rear legs are in contact."""
        rear_on = contacts[self.REAR_LEFT] or contacts[self.REAR_RIGHT]
        front_on = contacts[self.FRONT_LEFT] or contacts[self.FRONT_RIGHT]
        return rear_on and front_on
    
    def _get_stance_state(self, contacts):
        """Classify the current stance state."""
        if self._is_flight_phase(contacts):
            return 'flight'
        elif self._is_hind_stance(contacts):
            return 'hind'
        elif self._is_fore_stance(contacts):
            return 'fore'
        else:
            return 'mixed'
    
    def _find_flight_phases(self, contacts_list):
        """Find all flight phases in the contact sequence."""
        flight_phases = []
        in_flight = False
        flight_start = None
        
        for i, contacts in enumerate(contacts_list):
            is_flight = self._is_flight_phase(contacts)
            
            if is_flight and not in_flight:
                in_flight = True
                flight_start = i
            elif not is_flight and in_flight:
                in_flight = False
                flight_phases.append((flight_start, i))
        
        if in_flight and flight_start is not None:
            flight_phases.append((flight_start, len(contacts_list)))
            
        return flight_phases
    
    def _get_state_sequence(self, contacts_list):
        """Convert contact sequence to state sequence with transitions."""
        if not contacts_list:
            return []
        
        state_sequence = []
        current_state = self._get_stance_state(contacts_list[0])
        state_start = 0
        
        for i, contacts in enumerate(contacts_list[1:], 1):
            new_state = self._get_stance_state(contacts)
            if new_state != current_state:
                state_sequence.append((current_state, state_start, i))
                current_state = new_state
                state_start = i
        
        state_sequence.append((current_state, state_start, len(contacts_list)))
        return state_sequence
    
    def _detect_stride(self):
        """Detect if a valid stride pattern exists in the buffer."""
        buffer_len = len(self.contact_buffer)
        
        if buffer_len < self.min_stride_samples:
            return None
        
        # Ensure search_start_idx is valid
        if self.search_start_idx >= buffer_len:
            self.search_start_idx = max(0, buffer_len - self.window_size)
        
        # Get search window
        search_contacts = self.contact_buffer[self.search_start_idx:]
        search_times = self.time_buffer[self.search_start_idx:]
        
        if len(search_contacts) < self.min_stride_samples:
            return None
        
        # Get flight phases in search window
        flight_phases = self._find_flight_phases(search_contacts)
        
        # Detect based on gait class
        stride_info = None
        
        if self.gait_class == 'g0':
            stride_info = self._detect_g0_stride(search_contacts, search_times, flight_phases)
        elif self.gait_class == 'g2':
            stride_info = self._detect_g2_stride(search_contacts, search_times, flight_phases)
        elif self.gait_class == 'gg':
            stride_info = self._detect_gg_stride(search_contacts, search_times, flight_phases)
        elif self.gait_class == 'ge':
            stride_info = self._detect_ge_stride(search_contacts, search_times, flight_phases)
        
        if stride_info and stride_info['valid']:
            # Update search start for next detection (relative to current buffer)
            self.search_start_idx += stride_info['local_end_idx']
            self.stride_count += 1
            stride_info['stride_number'] = self.stride_count
            return stride_info
        
        return None
    
    def _detect_g0_stride(self, contacts, times, flight_phases):
        """Detect G0 stride: No flight phases."""
        # G0 should have NO flight phases
        if flight_phases:
            return None
        
        state_seq = self._get_state_sequence(contacts)
        
        if len(state_seq) < 4:
            return None
        
        for i in range(len(state_seq) - 3):
            window = state_seq[i:i+4]
            states = [s[0] for s in window]
            
            if 'flight' in states:
                continue
            
            unique_states = set(states)
            if len(unique_states) >= 2 and 'mixed' in unique_states:
                start_idx = window[0][1]
                end_idx = window[-1][2]
                
                if end_idx - start_idx >= self.min_stride_samples:
                    return {
                        'local_start_idx': start_idx,
                        'local_end_idx': end_idx,
                        'start_time': times[start_idx],
                        'end_time': times[end_idx - 1] if end_idx <= len(times) else times[-1],
                        'gait_type': self.gait_type,
                        'gait_class': 'G0',
                        'num_flight_phases': 0,
                        'valid': True
                    }
        
        return None
    
    def _detect_g2_stride(self, contacts, times, flight_phases):
        """Detect G2 stride: Two flight phases per stride."""
        if len(flight_phases) < 2:
            return None
        
        for i in range(len(flight_phases) - 1):
            fp1_start, fp1_end = flight_phases[i]
            fp2_start, fp2_end = flight_phases[i + 1]
            
            if fp2_start <= fp1_end:
                continue
            
            has_stance_between = False
            for j in range(fp1_end, fp2_start):
                if any(contacts[j]):
                    has_stance_between = True
                    break
            
            if not has_stance_between:
                continue
            
            start_idx = max(0, fp1_start - 3)
            end_idx = min(len(contacts), fp2_end + 3)
            
            if end_idx - start_idx >= self.min_stride_samples:
                return {
                    'local_start_idx': start_idx,
                    'local_end_idx': end_idx,
                    'start_time': times[start_idx],
                    'end_time': times[end_idx - 1] if end_idx <= len(times) else times[-1],
                    'gait_type': self.gait_type,
                    'gait_class': 'G2',
                    'num_flight_phases': 2,
                    'flight_phases': [(fp1_start, fp1_end), (fp2_start, fp2_end)],
                    'valid': True
                }
        
        return None
    
    def _detect_gg_stride(self, contacts, times, flight_phases):
        """Detect GG stride: One gathered flight phase."""
        if len(flight_phases) < 1:
            return None
        
        for fp_start, fp_end in flight_phases:
            start_idx = max(0, fp_start - 10)
            end_idx = min(len(contacts), fp_end + 10)
            
            window_contacts = contacts[start_idx:end_idx]
            window_flights = self._find_flight_phases(window_contacts)
            
            if len(window_flights) == 1 and (end_idx - start_idx) >= self.min_stride_samples:
                return {
                    'local_start_idx': start_idx,
                    'local_end_idx': end_idx,
                    'start_time': times[start_idx],
                    'end_time': times[end_idx - 1] if end_idx <= len(times) else times[-1],
                    'gait_type': self.gait_type,
                    'gait_class': 'GG',
                    'num_flight_phases': 1,
                    'valid': True
                }
        
        return None
    
    def _detect_ge_stride(self, contacts, times, flight_phases):
        """Detect GE stride: One extended flight phase."""
        return self._detect_gg_stride(contacts, times, flight_phases)
    
    def reset(self):
        """Reset the detector state."""
        self.contact_buffer = []
        self.time_buffer = []
        self.search_start_idx = 0
        self.stride_count = 0

class RealtimePlotter:
    """Lightweight real-time plotter for simulation data with minimal overhead."""

    def __init__(self, max_points=1000, update_interval=0.05, initial_offset=None, swing_ratio=None, 
                 contact_window_duration=1.0):
        """
        Args:
            max_points: Maximum number of points to display
            update_interval: Update plot every N seconds (reduces overhead)
            initial_offset: List of 4 phase offsets [FR, FL, RL, RR] - may be in radians
            swing_ratio: List of 4 swing ratios [FR, FL, RL, RR]
            contact_window_duration: Duration of time window to show in contacts plot (seconds)
        """
        self.max_points = max_points
        self.update_interval = update_interval
        self.last_update_time = 0
        self.contact_window_duration = contact_window_duration  # How many seconds to show
        
        # Store gait parameters (config order: [FR, FL, RL, RR])
        self.initial_offset = initial_offset if initial_offset is not None else [0, 0.1, 0.57, 0.47]
        self.swing_ratio = swing_ratio if swing_ratio is not None else [0.8, 0.8, 0.8, 0.8]
        
        # Config array order: [FR, FL, RL, RR] -> indices 0,1,2,3
        # Display y-positions: Rear Left=3, Front Left=2, Front Right=1, Rear Right=0
        self.config_idx_to_y = {0: 1, 1: 2, 2: 3, 3: 0}  # FR->1, FL->2, RL->3, RR->0
        
        # Contact array order: [FR, FL, RR, RL] -> indices 0,1,2,3
        # Display y-positions: Rear Left=3, Front Left=2, Front Right=1, Rear Right=0
        self.contact_idx_to_y = {0: 1, 1: 2, 2: 0, 3: 3}  # FR->1, FL->2, RR->0, RL->3

        # Stride event times for vertical separators in contacts plot
        self.stride_event_times = deque(maxlen=100)  # Store recent stride times

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
            'contacts': deque(maxlen=max_points)
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
        ax = self.axes[1, 0]
        ax.set_xlabel('Phase (normalized)')
        ax.set_ylabel('Leg')
        ax.set_title('Expected Gait Pattern (Blue=Stance)')
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 3.5)
        
        # Display order (top to bottom): Rear Left (y=3), Front Left (y=2), Front Right (y=1), Rear Right (y=0)
        leg_names = ['Rear Right', 'Front Right', 'Front Left', 'Rear Left']
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(leg_names)
        ax.grid(True, alpha=0.3, axis='x')

        bar_height = 0.6
        for config_idx, (offset_raw, swing) in enumerate(zip(self.initial_offset, self.swing_ratio)):
            y_pos = self.config_idx_to_y[config_idx]
            
            # Convert offset from radians to normalized [0, 1] if needed
            # If offset is > 1.0, assume it's in radians (was multiplied by 2*pi)
            if abs(offset_raw) > 1.0:
                offset = (offset_raw / (2 * np.pi)) % 1.0
            else:
                offset = offset_raw % 1.0
            
            stance_duration = 1.0 - swing
            stance_start = offset
            
            # Debug print
            leg_name = ['FR', 'FL', 'RL', 'RR'][config_idx]
            print(f"Gait plot - {leg_name}: offset_raw={offset_raw:.3f}, offset_norm={offset:.3f}, "
                  f"swing={swing:.3f}, stance_dur={stance_duration:.3f}, y_pos={y_pos}")
            
            # Handle wrap-around
            if stance_start + stance_duration <= 1.0:
                ax.barh(y_pos, stance_duration, left=stance_start, height=bar_height,
                       color='#4BACC6', edgecolor='black', linewidth=0.5)
            else:
                # First part: from stance_start to 1.0
                first_part_width = 1.0 - stance_start
                ax.barh(y_pos, first_part_width, left=stance_start, height=bar_height,
                       color='#4BACC6', edgecolor='black', linewidth=0.5)
                # Second part: from 0 to remainder
                second_part_width = stance_duration - first_part_width
                ax.barh(y_pos, second_part_width, left=0, height=bar_height,
                       color='#4BACC6', edgecolor='black', linewidth=0.5)

    def _setup_realtime_contacts_plot(self):
        """Setup real-time foot contact visualization."""
        ax = self.axes[1, 1]
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Leg')
        ax.set_title('Real-time Foot Contacts (Blue=Contact)')
        ax.set_ylim(-0.5, 3.5)
        
        # Same display order as gait pattern plot
        leg_names = ['Rear Right', 'Front Right', 'Front Left', 'Rear Left']
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(leg_names)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, self.contact_window_duration)

        self.contact_scatters = [ax.scatter([], [], c='#4BACC6', marker='s', s=80) for _ in range(4)]
        
        # Pre-allocate a fixed number of stride separator lines (reuse them)
        self.max_visible_stride_lines = 20  # Maximum lines we'll ever show at once
        self.contact_stride_lines = []
        for _ in range(self.max_visible_stride_lines):
            line = ax.axvline(x=-100, color='red', linestyle='--', linewidth=1.5, alpha=0.7, visible=False)
            self.contact_stride_lines.append(line)

    def add_stride_event(self, time_val):
        """Record a stride event time for visualization."""
        self.stride_event_times.append(time_val)

    def _update_contacts_plot(self):
        """Update the real-time contacts visualization."""
        if len(self.contact_history['time']) == 0:
            return

        ax = self.axes[1, 1]
        times = list(self.contact_history['time'])
        contacts = list(self.contact_history['contacts'])
        
        # Calculate visible time window
        max_time = max(times)
        window_start = max(0, max_time - self.contact_window_duration)
        window_end = max_time + 0.1  # Small padding

        # Filter data to only show within window
        visible_indices = [i for i, t in enumerate(times) if t >= window_start]
        
        # Contact array order: [FR, FL, RR, RL] -> indices 0,1,2,3
        for contact_idx in range(4):
            y_pos = self.contact_idx_to_y[contact_idx]
            
            # Only include contacts within the visible window
            contact_times = [times[i] for i in visible_indices if contacts[i][contact_idx]]
            contact_y = [y_pos] * len(contact_times)

            if contact_times:
                self.contact_scatters[contact_idx].set_offsets(np.c_[contact_times, contact_y])
            else:
                self.contact_scatters[contact_idx].set_offsets(np.empty((0, 2)))

        # Update x-axis limits to sliding window
        ax.set_xlim(window_start, window_end)
        
        # Update stride separator lines
        self._update_contact_stride_lines(window_start, window_end)

    def _update_contact_stride_lines(self, window_start, window_end):
        """Update vertical stride separator lines in contacts plot by reusing existing line objects."""
        # Get stride times within visible window
        visible_stride_times = [t for t in self.stride_event_times if window_start <= t <= window_end]
        
        # Update pre-allocated lines
        for i, line in enumerate(self.contact_stride_lines):
            if i < len(visible_stride_times):
                # Position this line at the stride time and make visible
                line.set_xdata([visible_stride_times[i], visible_stride_times[i]])
                line.set_visible(True)
            else:
                # Hide unused lines
                line.set_visible(False)

    def add_contact_data(self, time_val, contacts):
        """Add foot contact data point."""
        self.contact_history['time'].append(time_val)
        self.contact_history['contacts'].append(list(contacts))

    def add_data(self, time_val, desired_vel=None, velocity=None, stride_length=None, is_stride_event=False):
        """Add data point(s) to the plots."""
        if velocity is not None:
            self.velocity_data['x'].append(time_val)
            self.velocity_data['y'].append(velocity)

        if desired_vel is not None:
            self.desired_vel_data['x'].append(time_val)
            self.desired_vel_data['y'].append(desired_vel)

        if stride_length is not None:
            self.stride_data['x'].append(time_val)
            self.stride_data['y'].append(stride_length)

        if is_stride_event:
            self.stride_markers['x'].append(time_val)
            self.stride_markers['y'].append(0)
            self.stride_event_times.append(time_val)  # Also record for contacts plot

        current_real_time = time.time()
        if (current_real_time - self.last_update_time) >= self.update_interval:
            self._update_plot()
            self.last_update_time = current_real_time

    def add_cot_data(self, velocity, cot):
        """Add COT vs velocity data point."""
        self.cot_velocity_data['x'].append(velocity)
        self.cot_velocity_data['y'].append(cot)

    def _update_plot(self):
        """Update the plots efficiently."""
        try:
            ax1 = self.axes[0, 0]

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

            if len(self.cot_velocity_data['x']) > 0:
                self.cot_scatter.set_offsets(
                    np.c_[list(self.cot_velocity_data['x']), list(self.cot_velocity_data['y'])]
                )
                self._auto_scale_cot_plot(self.axes[0, 1], self.cot_velocity_data)

            self._update_contacts_plot()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception as e:
            print(f"Plot update error: {e}")

    def _setup_combined_plot(self):
        """Setup combined velocity and stride length plot with dual y-axes."""
        ax1 = self.axes[0, 0]
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
        ax = self.axes[0, 1]
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

    def close(self):
        """Close the plot."""
        plt.close(self.fig)


from src.envs import env_wrappers
torch.set_printoptions(precision=2, sci_mode=False)

flags.DEFINE_string("logdir", None, "logdir.")
flags.DEFINE_bool("use_gpu", False, "whether to use GPU.")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_bool("use_real_robot", False, "whether to use real robot.")
flags.DEFINE_integer("num_envs", 1, "number of environments to evaluate in parallel.")
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
    del argv

    device = "cuda" if FLAGS.use_gpu else "cpu"

    # Load config and policy
    if FLAGS.logdir.endswith("pt"):
        config_path = os.path.join(os.path.dirname(FLAGS.logdir), "config.yaml")
        policy_path = FLAGS.logdir
        root_path = os.path.dirname(FLAGS.logdir)
    else:
        config_path = os.path.join(FLAGS.logdir, "config.yaml")
        policy_path = get_latest_policy_path(FLAGS.logdir)
        root_path = FLAGS.logdir

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    with config.unlocked():
        velocity_up = torch.linspace(0.8, 4.0, 50)
        velocity_schedule = velocity_up
        config.environment.jumping_distance_schedule = velocity_schedule / config.environment.gait.stepping_frequency
        config.environment.gait.desired_velocity = torch.tensor([velocity_schedule[0].item(), 0, 0])
        config.environment.max_jumps = 3000

    env = config.env_class(num_envs=FLAGS.num_envs,
                           device=device,
                           config=config.environment,
                           show_gui=FLAGS.show_gui,
                           use_real_robot=FLAGS.use_real_robot)
    env = env_wrappers.RangeNormalize(env)
    
    if FLAGS.use_real_robot:
        env.robot.state_estimator.use_external_contact_estimator = (not FLAGS.use_contact_sensor)

    runner = OnPolicyRunner(env, config.training, policy_path, device=device)
    runner.load(policy_path)
    policy = runner.get_inference_policy()
    runner.alg.actor_critic.train()

    state, _ = env.reset()
    total_reward = torch.zeros(FLAGS.num_envs, device=device)
    steps_count = 0

    start_time = time.time()
    logs = []

    # Get gait parameters
    initial_offset = list(config.environment.gait.initial_offset)
    swing_ratio = list(config.environment.gait.swing_ratio)
    gait_name = getattr(config.environment.gait, 'gait_name', 'g2_transverse')
    
    print(f"Gait: {gait_name}")
    print(f"Initial offset: {initial_offset}")
    print(f"Swing ratio: {swing_ratio}")

    # Initialize plotter
    plotter = None
    if FLAGS.enable_plotting:
        plotter = RealtimePlotter(
            max_points=10000,
            update_interval=FLAGS.plot_update_interval,
            initial_offset=initial_offset,
            swing_ratio=swing_ratio,
            contact_window_duration=0.5
        )
        print(f"Real-time plotting enabled (update every {FLAGS.plot_update_interval}s)")

    # Initialize stride detector
    stride_detector = GaitStrideDetector(gait_type=gait_name, window_size=200)
    print(f"Stride detector initialized for gait type: {gait_name}")

    # COT data storage
    max_strides = 10000
    cot_data_arrays = {
        'velocity': np.zeros(max_strides, dtype=np.float32),
        'cot': np.zeros(max_strides, dtype=np.float32),
    }
    cot_data_count = 0

    # Pre-allocate buffers for stride data
    max_stride_steps = 1000
    num_joints = 12
    stride_torques = np.zeros((num_joints, max_stride_steps), dtype=np.float32)
    stride_velocities = np.zeros((num_joints, max_stride_steps), dtype=np.float32)
    stride_step_count = 0
    stride_x_velocities = []
    stride_time_start = 0.0

    print("Starting simulation loop...")

    velocity_index = 0
    with torch.inference_mode():
        while True:
            steps_count += 1
            action = policy(state)
            state, _, reward, done, info = env.step(action)

            # Extract current data
            current_time = env.robot.time_since_reset.item()
            velocity = torch.norm(env.robot.base_vel).item()
            current_contacts = env.robot.foot_contacts[0].tolist()

            body_frame_velocity = env.robot.base_velocity_body_frame[0]
            forward_velocity = body_frame_velocity[0].item()

            current_joint_torques = env.robot.motor_torques[0].cpu().numpy()
            current_joint_velocities = env.robot.motor_velocities[0].cpu().numpy()

            # Store stride data
            if stride_step_count < max_stride_steps:
                stride_torques[:, stride_step_count] = current_joint_torques
                stride_velocities[:, stride_step_count] = current_joint_velocities
                stride_x_velocities.append(forward_velocity)
                stride_step_count += 1

            stride_length = env._jumping_distance[0, 0].item() if torch.is_tensor(env._jumping_distance[0]) else float(env._jumping_distance)

            # Add contact data to plotter
            if plotter is not None:
                plotter.add_contact_data(current_time, current_contacts)

            # Update velocity schedule Independently of stride detection
            velocity_update_interval = 50
            if steps_count % velocity_update_interval == 0:
                if velocity_index < len(velocity_schedule) - 1:
                    velocity_index += 1
                env._desired_velocity[:, 0] = velocity_schedule[velocity_index]
                env._desired_velocity[:, 1:] = 0

            # Use stride detector
            stride_info = stride_detector.add_contact(current_time, current_contacts)

            is_stride_event = False
            if stride_info is not None and stride_info['valid']:
                is_stride_event = True
                print(f">>> STRIDE #{stride_info['stride_number']} ({stride_info['gait_class']}) "
                      f"detected at t={current_time:.2f}s, "
                      f"flight_phases={stride_info['num_flight_phases']} <<<")

                # Calculate COT for completed stride
                if stride_step_count > 0 and len(stride_x_velocities) > 0:
                    avg_forward_velocity = np.mean(stride_x_velocities)
                    stride_duration = current_time - stride_time_start

                    if stride_duration > 0:
                        stride_velocity, stride_cot = calculate_cot_per_stride2(
                            stride_torques[:, :stride_step_count],
                            stride_velocities[:, :stride_step_count],
                            env._config.env_dt,
                            avg_forward_velocity,
                            stride_duration,
                            env.robot.mass
                        )

                        print(f"  Stride Vel: {stride_velocity:.3f} m/s, COT: {stride_cot:.4f}")

                        # Filter valid COT values
                        if 0 < stride_cot < 5.0:
                            if cot_data_count < max_strides:
                                cot_data_arrays['velocity'][cot_data_count] = stride_velocity
                                cot_data_arrays['cot'][cot_data_count] = stride_cot
                                cot_data_count += 1

                            if plotter is not None:
                                plotter.add_cot_data(stride_velocity, stride_cot)

                # Reset for next stride
                stride_step_count = 0
                stride_x_velocities = []
                stride_time_start = current_time

            # Update plotter with velocity/stride data
            if plotter is not None:
                plotter.add_data(
                    time_val=current_time,
                    desired_vel=velocity_schedule[min(velocity_index, len(velocity_schedule)-1)].item(),
                    velocity=velocity,
                    stride_length=stride_length,
                    is_stride_event=is_stride_event
                )

            if steps_count >= 1000:  # break from loop and save
                break

    # Cleanup code...
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"\n{'='*60}")
    print(f"Simulation Complete!")
    print(f"{'='*60}")
    print(f"Total steps: {steps_count}")
    print(f"Time elapsed: {elapsed:.2f}s")
    print(f"Steps per second: {steps_count / elapsed:.1f} Hz")
    print(f"Total strides detected: {stride_detector.stride_count}")
    print(f"COT samples recorded: {cot_data_count}")
    print(f"{'='*60}\n")

    # Save COT data
    if cot_data_count > 0:
        cot_output_path = os.path.join(root_path, f"cot_data_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv")
        data_to_save = np.column_stack([
            cot_data_arrays['velocity'][:cot_data_count],
            cot_data_arrays['cot'][:cot_data_count],
        ])
        np.savetxt(cot_output_path, data_to_save,
                   delimiter=',',
                   header='velocity,cot',
                   comments='',
                   fmt='%.6f')
        print(f"COT data saved to: {cot_output_path}")

    if plotter is not None:
        print("Plot window is open. Press Enter to close and exit...")
        input()
        plotter.close()


if __name__ == "__main__":
    app.run(main)
