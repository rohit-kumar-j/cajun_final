"""
Interactive Raibert Foot Placement Calculator - Side View with Bezier Curves
Fixed version with better layout and visibility
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.lines import Line2D
import matplotlib.animation as animation


def cubic_bezier(x0, x1, t):
    """Cubic bezier interpolation (matches your controller)."""
    progress = t**3 + 3 * t**2 * (1 - t)
    return x0 + progress * (x1 - x0)


def gen_swing_trajectory(phase, start_pos, mid_pos, end_pos):
    """Generate swing foot trajectory using Bezier curves."""
    cutoff = 0.5
    if phase < cutoff:
        return cubic_bezier(start_pos, mid_pos, phase / cutoff)
    else:
        return cubic_bezier(mid_pos, end_pos, (phase - cutoff) / (1 - cutoff))


class RaibertSideViewCalculator:
    def __init__(self):
        # Parameters
        self.params = {
            'frequency': 1.5,
            'swing_ratio': 0.6,
            'raibert_kp': 0.8,
            'v_current': 4.0,
            'delta_v': 1.0,        # delta_v = v_desired - v_current (positive = want to go faster)
            'foot_height': 0.12,
            'land_position': -0.2667,  # Direct control of landing position
        }
        
        # Lock states for each parameter
        self.locks = {key: False for key in self.params}
        
        # Flag to prevent circular updates
        self._updating = False
        
        # Robot dimensions
        self.body_length = 0.4
        self.body_height = 0.1
        self.base_height = 0.28
        
        # Animation state
        self.phase = 0.0
        self.animation_running = True
        
        self.setup_figure()
        
    def calculate(self, from_land_pos=False):
        """Calculate Raibert foot placement - NO CLIPPING.
        
        If from_land_pos=True, back-calculate delta_v from the land_position.
        """
        freq = self.params['frequency']
        swing_ratio = self.params['swing_ratio']
        kp = self.params['raibert_kp']
        v_curr = self.params['v_current']
        
        stance_ratio = 1.0 - swing_ratio
        cycle_period = 1.0 / freq
        stance_duration = stance_ratio * cycle_period
        swing_duration = swing_ratio * cycle_period
        
        # Baseline is always the same
        baseline = v_curr * stance_duration / 2.0
        
        if from_land_pos:
            # Back-calculate from land_position
            land_position = self.params['land_position']
            # land_pos = baseline + kp * (-delta_v)
            # correction = land_pos - baseline
            # -delta_v = correction / kp
            # delta_v = -correction / kp
            correction = land_position - baseline
            if abs(kp) > 0.01:
                delta_v = -correction / kp
            else:
                delta_v = 0
            v_error = -delta_v
            v_des = v_curr + delta_v
        else:
            delta_v = self.params['delta_v']
            v_des = v_curr + delta_v
            v_error = -delta_v
            correction = kp * v_error
            land_position = baseline + correction
        
        return {
            'stance_duration': stance_duration,
            'swing_duration': swing_duration,
            'cycle_period': cycle_period,
            'baseline': baseline,
            'correction': correction,
            'land_position': land_position,
            'v_error': v_error,
            'v_desired': v_des,
            'delta_v': delta_v,
        }
    
    def setup_figure(self):
        """Create figure with clean layout."""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor('#1a1a2e')
        
        # Title
        self.fig.text(0.35, 0.96, 'Raibert Foot Placement Calculator - Side View', 
                     fontsize=20, fontweight='bold', color='white', ha='center')
        
        # Main robot view (left side, large)
        self.ax_robot = self.fig.add_axes([0.03, 0.32, 0.62, 0.6])
        self.ax_robot.set_facecolor('#0d1b2a')
        
        # Formula panel (below robot view)
        self.ax_formula = self.fig.add_axes([0.03, 0.03, 0.62, 0.26])
        self.ax_formula.set_facecolor('#0d1b2a')
        self.ax_formula.axis('off')
        
        # Sliders on the right side - all stacked vertically with labels above
        self.setup_sliders()
        
        # Draw initial state
        self.draw_formula()
        self.draw_robot()
        
        # Animation
        self.anim = animation.FuncAnimation(
            self.fig, self.animate, interval=33, blit=False, cache_frame_data=False)
    
    def setup_sliders(self):
        """Create sliders on the right side with labels above and lock buttons."""
        self.sliders = {}
        self.lock_buttons = {}
        self.value_texts = {}  # Store value text objects
        
        # Slider configurations: (name, label, min, max, initial)
        slider_configs = [
            ('frequency', 'Freq (Hz)', 0.5, 4.0, 1.5),
            ('swing_ratio', 'Swing Ratio', 0.3, 0.9, 0.6),
            ('raibert_kp', 'Kp', 0.1, 2.5, 0.8),
            ('v_current', 'V_curr (m/s)', 0.0, 12.0, 4.0),
            ('delta_v', 'Delta_V', -6.0, 6.0, 1.0),
            ('land_position', 'LAND (m)', -2.0, 2.0, -0.2667),
            ('foot_height', 'Foot_H (m)', 0.05, 0.35, 0.12),
        ]
        
        # Layout parameters
        slider_left = 0.68
        slider_width = 0.18
        lock_btn_width = 0.035
        slider_height = 0.018
        start_y = 0.91
        spacing = 0.068
        
        for i, (name, label, vmin, vmax, vinit) in enumerate(slider_configs):
            y_pos = start_y - i * spacing
            
            # Label with value above slider
            if name == 'land_position':
                label_color = '#ffeb3b'
            else:
                label_color = 'white'
            
            # Create label text (will show name and value)
            label_text = self.fig.text(slider_left, y_pos + 0.025, 
                                       f'{label}: {vinit:.2f}',
                                       fontsize=9, color=label_color, ha='left', fontweight='bold')
            self.value_texts[name] = label_text
            
            # Slider
            ax_slider = self.fig.add_axes([slider_left, y_pos, slider_width, slider_height])
            ax_slider.set_facecolor('#2d3a4a')
            
            slider_color = '#ffeb3b' if name == 'land_position' else '#4fc3f7'
            
            slider = Slider(
                ax_slider, '', vmin, vmax,
                valinit=vinit,
                color=slider_color,
                valstep=0.001 if name in ['land_position', 'swing_ratio'] else 0.01
            )
            
            slider.valtext.set_visible(False)  # Hide default value text
            
            slider.on_changed(self.make_slider_callback(name))
            self.sliders[name] = slider
            
            # Lock button (to the right of slider)
            ax_lock = self.fig.add_axes([slider_left + slider_width + 0.01, y_pos, lock_btn_width, slider_height])
            btn_lock = Button(ax_lock, 'L', color='#2d3a4a', hovercolor='#4a5a6a')
            btn_lock.label.set_color('white')
            btn_lock.label.set_fontsize(9)
            btn_lock.on_clicked(self.make_lock_callback(name, ax_lock))
            self.lock_buttons[name] = (btn_lock, ax_lock)
        
        # Buttons at the bottom right
        btn_y = 0.02
        ax_pause = self.fig.add_axes([0.68, btn_y, 0.07, 0.035])
        self.btn_pause = Button(ax_pause, 'Pause', color='#2d3a4a', hovercolor='#4a5a6a')
        self.btn_pause.label.set_color('white')
        self.btn_pause.label.set_fontsize(9)
        self.btn_pause.on_clicked(self.toggle_animation)
        
        ax_reset = self.fig.add_axes([0.76, btn_y, 0.07, 0.035])
        self.btn_reset = Button(ax_reset, 'Reset', color='#2d3a4a', hovercolor='#4a5a6a')
        self.btn_reset.label.set_color('white')
        self.btn_reset.label.set_fontsize(9)
        self.btn_reset.on_clicked(self.reset_params)
        
        ax_print = self.fig.add_axes([0.84, btn_y, 0.07, 0.035])
        self.btn_print = Button(ax_print, 'PRINT', color='#1b5e20', hovercolor='#2e7d32')
        self.btn_print.label.set_color('white')
        self.btn_print.label.set_fontsize(9)
        self.btn_print.label.set_fontweight('bold')
        self.btn_print.on_clicked(self.print_debug)
        
        ax_unlock = self.fig.add_axes([0.92, btn_y, 0.05, 0.035])
        self.btn_unlock = Button(ax_unlock, 'UnL', color='#5d4037', hovercolor='#795548')
        self.btn_unlock.label.set_color('white')
        self.btn_unlock.label.set_fontsize(8)
        self.btn_unlock.on_clicked(self.unlock_all)
    
    def _update_value_texts(self):
        """Update all value text labels to show current values."""
        labels = {
            'frequency': 'Freq (Hz)',
            'swing_ratio': 'Swing Ratio', 
            'raibert_kp': 'Kp',
            'v_current': 'V_curr (m/s)',
            'delta_v': 'Delta_V',
            'land_position': 'LAND (m)',
            'foot_height': 'Foot_H (m)',
        }
        for name, label in labels.items():
            if name in self.value_texts:
                val = self.params[name]
                lock_indicator = " 🔒" if self.locks[name] else ""
                self.value_texts[name].set_text(f'{label}: {val:.3f}{lock_indicator}')
    
    def make_lock_callback(self, name, ax_lock):
        """Create a callback for lock button."""
        def callback(event):
            self.locks[name] = not self.locks[name]
            btn, ax = self.lock_buttons[name]
            if self.locks[name]:
                btn.label.set_text('🔒')
                ax.set_facecolor('#c62828')  # Red when locked
                self.sliders[name].poly.set_facecolor('#666666')  # Gray out slider
            else:
                btn.label.set_text('L')
                ax.set_facecolor('#2d3a4a')
                # Restore original color
                if name == 'land_position':
                    self.sliders[name].poly.set_facecolor('#ffeb3b')
                else:
                    self.sliders[name].poly.set_facecolor('#4fc3f7')
            self._update_value_texts()
            self.fig.canvas.draw_idle()
        return callback
    
    def unlock_all(self, event):
        """Unlock all parameters."""
        for name in self.locks:
            self.locks[name] = False
            btn, ax = self.lock_buttons[name]
            btn.label.set_text('L')
            ax.set_facecolor('#2d3a4a')
            if name == 'land_position':
                self.sliders[name].poly.set_facecolor('#ffeb3b')
            else:
                self.sliders[name].poly.set_facecolor('#4fc3f7')
        self.fig.canvas.draw_idle()
    
    def make_slider_callback(self, name):
        def callback(val):
            if self._updating:
                return
            
            # If this parameter is locked, revert the change
            if self.locks[name]:
                self.sliders[name].set_val(self.params[name])
                return
                
            self._updating = True
            self.params[name] = val
            
            # Now solve for unlocked parameters to maintain consistency
            # The Raibert equation: land_pos = baseline + correction
            # where: baseline = v_curr * T_stance / 2
            #        correction = Kp * (-delta_v)
            #        T_stance = (1 - swing_ratio) / frequency
            
            # Determine which parameters can be adjusted
            adjustable = [p for p in ['frequency', 'swing_ratio', 'raibert_kp', 
                                       'delta_v', 'land_position'] 
                         if not self.locks[p] and p != name]
            
            if name == 'land_position':
                # User set land_position, solve for another parameter
                self._solve_for_unlocked(adjustable, 'land_position')
            elif name in ['frequency', 'swing_ratio', 'raibert_kp', 'delta_v', 'v_current']:
                # User changed an input, update land_position if unlocked
                if not self.locks['land_position']:
                    results = self.calculate(from_land_pos=False)
                    self.params['land_position'] = results['land_position']
                    clamped = max(-2.0, min(2.0, results['land_position']))
                    self.sliders['land_position'].set_val(clamped)
                else:
                    # land_position is locked, solve for something else
                    self._solve_for_unlocked(adjustable, name)
            
            self._updating = False
            self._update_value_texts()
            self.draw_formula()
            
        return callback
    
    def _solve_for_unlocked(self, adjustable, changed_param):
        """Solve for an unlocked parameter to satisfy the Raibert equation."""
        # Priority order for what to adjust
        priority = ['delta_v', 'raibert_kp', 'frequency', 'swing_ratio']
        
        target_land_pos = self.params['land_position']
        v_curr = self.params['v_current']
        freq = self.params['frequency']
        swing_ratio = self.params['swing_ratio']
        kp = self.params['raibert_kp']
        
        # Calculate current baseline
        stance_ratio = 1.0 - swing_ratio
        T_stance = stance_ratio / freq
        baseline = v_curr * T_stance / 2.0
        
        # Required correction
        required_correction = target_land_pos - baseline
        
        for param in priority:
            if param in adjustable:
                if param == 'delta_v':
                    # correction = Kp * (-delta_v) => delta_v = -correction / Kp
                    if abs(kp) > 0.01:
                        new_delta_v = -required_correction / kp
                        new_delta_v = max(-6.0, min(6.0, new_delta_v))
                        self.params['delta_v'] = new_delta_v
                        self.sliders['delta_v'].set_val(new_delta_v)
                        return
                        
                elif param == 'raibert_kp':
                    # correction = Kp * (-delta_v) => Kp = correction / (-delta_v)
                    delta_v = self.params['delta_v']
                    if abs(delta_v) > 0.01:
                        new_kp = required_correction / (-delta_v)
                        new_kp = max(0.1, min(2.5, new_kp))
                        self.params['raibert_kp'] = new_kp
                        self.sliders['raibert_kp'].set_val(new_kp)
                        return
                        
                elif param == 'frequency':
                    # baseline = v_curr * (1-swing_ratio) / (2*freq)
                    # We need: target_land_pos = baseline + Kp*(-delta_v)
                    # So: baseline = target_land_pos - Kp*(-delta_v)
                    delta_v = self.params['delta_v']
                    correction = kp * (-delta_v)
                    needed_baseline = target_land_pos - correction
                    # baseline = v_curr * (1-swing_ratio) / (2*freq)
                    # freq = v_curr * (1-swing_ratio) / (2*baseline)
                    if abs(needed_baseline) > 0.01 and v_curr > 0:
                        new_freq = v_curr * stance_ratio / (2 * needed_baseline)
                        new_freq = max(0.5, min(4.0, new_freq))
                        self.params['frequency'] = new_freq
                        self.sliders['frequency'].set_val(new_freq)
                        return
                        
                elif param == 'swing_ratio':
                    # Similar logic for swing_ratio
                    delta_v = self.params['delta_v']
                    correction = kp * (-delta_v)
                    needed_baseline = target_land_pos - correction
                    # baseline = v_curr * (1-swing_ratio) / (2*freq)
                    # (1-swing_ratio) = 2*freq*baseline / v_curr
                    # swing_ratio = 1 - 2*freq*baseline / v_curr
                    if v_curr > 0.01:
                        new_swing = 1.0 - (2 * freq * needed_baseline / v_curr)
                        new_swing = max(0.3, min(0.9, new_swing))
                        self.params['swing_ratio'] = new_swing
                        self.sliders['swing_ratio'].set_val(new_swing)
                        return
    
    def toggle_animation(self, event):
        self.animation_running = not self.animation_running
        self.btn_pause.label.set_text('Play' if not self.animation_running else 'Pause')
        self.fig.canvas.draw_idle()
    
    def reset_params(self, event):
        self._updating = True
        defaults = {
            'frequency': 1.5, 'swing_ratio': 0.6, 'raibert_kp': 0.8,
            'v_current': 4.0, 'delta_v': 1.0, 'foot_height': 0.12
        }
        for name, val in defaults.items():
            self.params[name] = val
            self.sliders[name].set_val(val)
        
        # Calculate and set land_position
        results = self.calculate(from_land_pos=False)
        self.params['land_position'] = results['land_position']
        self.sliders['land_position'].set_val(results['land_position'])
        self._updating = False
        self.draw_formula()
    
    def print_debug(self, event):
        """Print all debug data to console."""
        results = self.calculate()
        
        # Current phase info
        cycle_progress = self.phase % 1.0
        swing_ratio = self.params['swing_ratio']
        stance_ratio = 1.0 - swing_ratio
        
        # Front leg phase
        front_in_swing = cycle_progress >= stance_ratio
        if front_in_swing:
            front_swing_phase = (cycle_progress - stance_ratio) / swing_ratio
        else:
            front_swing_phase = None
            front_stance_progress = cycle_progress / stance_ratio
        
        # Rear leg phase
        rear_cycle = (cycle_progress + 0.5) % 1.0
        rear_in_swing = rear_cycle >= stance_ratio
        if rear_in_swing:
            rear_swing_phase = (rear_cycle - stance_ratio) / swing_ratio
        else:
            rear_swing_phase = None
            rear_stance_progress = rear_cycle / stance_ratio
        
        # Foot positions
        land_pos = results['land_position']
        swing_start_offset = -results['baseline']  # Match the drawing code
        
        front_hip_x = self.body_length / 4
        rear_hip_x = -self.body_length / 4
        
        front_start = front_hip_x + swing_start_offset
        front_land = front_hip_x + land_pos
        rear_start = rear_hip_x + swing_start_offset
        rear_land = rear_hip_x + land_pos
        
        if front_in_swing and front_swing_phase is not None:
            front_foot = self.get_foot_position(front_swing_phase, front_start, front_land)
        else:
            fp = front_stance_progress if not front_in_swing else 0
            stance_travel = front_land - front_start
            front_foot = np.array([front_land - fp * stance_travel, 0])
        
        if rear_in_swing and rear_swing_phase is not None:
            rear_foot = self.get_foot_position(rear_swing_phase, rear_start, rear_land)
        else:
            rp = rear_stance_progress if not rear_in_swing else 0
            stance_travel = rear_land - rear_start
            rear_foot = np.array([rear_land - rp * stance_travel, 0])
        
        print("\n" + "=" * 80)
        print("DEBUG OUTPUT - RAIBERT CALCULATOR")
        print("=" * 80)
        
        v_des = self.params['v_current'] + results['delta_v']
        
        print("\n--- INPUT PARAMETERS ---")
        lock_str = lambda n: " [LOCKED]" if self.locks[n] else ""
        print(f"  frequency:      {self.params['frequency']:.4f} Hz{lock_str('frequency')}")
        print(f"  swing_ratio:    {self.params['swing_ratio']:.4f}{lock_str('swing_ratio')}")
        print(f"  raibert_kp:     {self.params['raibert_kp']:.4f}{lock_str('raibert_kp')}")
        print(f"  v_current:      {self.params['v_current']:.4f} m/s{lock_str('v_current')}")
        print(f"  delta_v:        {self.params['delta_v']:+.4f} m/s{lock_str('delta_v')}")
        print(f"  land_position:  {self.params['land_position']:+.4f} m{lock_str('land_position')}")
        print(f"  foot_height:    {self.params['foot_height']:.4f} m{lock_str('foot_height')}")
        print(f"\n  COMPUTED:")
        print(f"  v_desired:      {v_des:.4f} m/s  (= v_current + delta_v)")
        
        print("\n--- CALCULATED TIMING ---")
        print(f"  cycle_period:   {results['cycle_period']:.4f} s")
        print(f"  stance_duration: {results['stance_duration']:.4f} s")
        print(f"  swing_duration:  {results['swing_duration']:.4f} s")
        print(f"  stance_ratio:    {stance_ratio:.4f}")
        
        print("\n--- RAIBERT CALCULATION ---")
        print(f"  delta_v = v_desired - v_current = {self.params['delta_v']:+.4f} m/s")
        print(f"  v_error = v_current - v_desired = -delta_v = {results['v_error']:+.4f} m/s")
        print()
        print(f"  baseline = v_current * T_stance / 2")
        print(f"           = {self.params['v_current']:.4f} * {results['stance_duration']:.4f} / 2")
        print(f"           = {results['baseline']:.4f} m")
        print()
        print(f"  correction = Kp * v_error = Kp * (-delta_v)")
        print(f"             = {self.params['raibert_kp']:.4f} * {results['v_error']:+.4f}")
        print(f"             = {results['correction']:.4f} m")
        print()
        print(f"  LAND_POSITION = baseline + correction")
        print(f"                = {results['baseline']:.4f} + {results['correction']:.4f}")
        print(f"                = {results['land_position']:.4f} m")
        print()
        print(f"  SWING_START = -baseline = {swing_start_offset:.4f} m")
        print(f"  (foot drifts back by baseline during stance, so swing starts there)")
        
        print("\n--- ANIMATION STATE ---")
        print(f"  raw_phase:      {self.phase:.4f}")
        print(f"  cycle_progress: {cycle_progress:.4f} (0-1)")
        print(f"  phase_percent:  {cycle_progress * 100:.1f}%")
        
        print("\n--- FRONT LEG ---")
        print(f"  in_swing:       {front_in_swing}")
        print(f"  swing_phase:    {front_swing_phase}")
        print(f"  trajectory:     start={front_start:.4f}, land={front_land:.4f}")
        print(f"  foot_position:  x={front_foot[0]:.4f}, y={front_foot[1]:.4f}")
        
        print("\n--- REAR LEG ---")
        print(f"  rear_cycle:     {rear_cycle:.4f}")
        print(f"  in_swing:       {rear_in_swing}")
        print(f"  swing_phase:    {rear_swing_phase}")
        print(f"  trajectory:     start={rear_start:.4f}, land={rear_land:.4f}")
        print(f"  foot_position:  x={rear_foot[0]:.4f}, y={rear_foot[1]:.4f}")
        
        print("\n--- BEZIER TRAJECTORY SAMPLE (front leg) ---")
        if front_in_swing and front_swing_phase is not None:
            print(f"  Bezier phase = {front_swing_phase:.4f}")
            for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
                pos = self.get_foot_position(t, front_start, front_land)
                print(f"    t={t:.2f}: x={pos[0]:.4f}, y={pos[1]:.4f}")
        else:
            print("  (front leg in stance)")
        
        print("\n" + "=" * 80)
        print("END DEBUG OUTPUT")
        print("=" * 80 + "\n")
    
    def draw_formula(self):
        """Draw formula and results panel."""
        self.ax_formula.clear()
        self.ax_formula.set_facecolor('#0d1b2a')
        self.ax_formula.axis('off')
        
        results = self.calculate()
        
        # Determine state
        v_err = results['v_error']
        if abs(v_err) < 0.1:
            state = "STEADY"
            state_color = '#4fc3f7'
        elif v_err > 0:
            state = "BRAKING"
            state_color = '#ef5350'
        else:
            state = "ACCELERATING"
            state_color = '#ffeb3b'
        
        # Large, readable formula text
        v_des = results['v_desired']
        delta_v = results['delta_v']  # Use calculated value
        
        formula_text = f"""RAIBERT FORMULA
================

  land_pos  =  (v_current x T_stance / 2)  +  (Kp x v_error)
            =  baseline  +  correction

  WHERE:
    T_stance  =  (1 - swing_ratio) / freq  =  (1 - {self.params['swing_ratio']:.2f}) / {self.params['frequency']:.2f}  =  {results['stance_duration']:.4f} s
    v_desired =  v_current + delta_v       =  {self.params['v_current']:.2f} + ({delta_v:+.2f})       =  {v_des:.2f} m/s
    v_error   =  -delta_v                  =  {results['v_error']:+.4f} m/s

CALCULATION
===========

  Baseline Term    =  v x T_stance / 2  =  {self.params['v_current']:.2f} x {results['stance_duration']:.4f} / 2  =  {results['baseline']:+.4f} m
  Correction Term  =  Kp x v_error      =  {self.params['raibert_kp']:.2f} x ({results['v_error']:+.4f})       =  {results['correction']:+.4f} m
  ───────────────────────────────────────────────────────────────────────────────────────
  LANDING POSITION =  {results['baseline']:+.4f} + ({results['correction']:+.4f})  =  {results['land_position']:+.4f} m   [{state}]

TIMING:  Cycle={results['cycle_period']:.3f}s | Stance={results['stance_duration']:.3f}s | Swing={results['swing_duration']:.3f}s

[YELLOW slider: Set land_pos directly -> back-calculates delta_v]"""

        self.ax_formula.text(0.02, 0.95, formula_text, 
                            transform=self.ax_formula.transAxes,
                            fontsize=11, color='white', family='monospace',
                            verticalalignment='top')
    
    def get_bezier_trajectory(self, start_x, land_x, num_points=60):
        """Generate full Bezier trajectory points for drawing the curve."""
        foot_height = self.params['foot_height']
        ground_y = 0
        mid_x = (start_x + land_x) / 2
        mid_y = foot_height
        
        start_pos = np.array([start_x, ground_y])
        mid_pos = np.array([mid_x, mid_y])
        end_pos = np.array([land_x, ground_y])
        
        phases = np.linspace(0, 1, num_points)
        trajectory = np.array([gen_swing_trajectory(p, start_pos, mid_pos, end_pos) 
                              for p in phases])
        return trajectory
    
    def get_foot_position(self, phase, start_x, land_x):
        """Get foot position at given phase."""
        foot_height = self.params['foot_height']
        ground_y = 0
        mid_x = (start_x + land_x) / 2
        mid_y = foot_height
        
        start_pos = np.array([start_x, ground_y])
        mid_pos = np.array([mid_x, mid_y])
        end_pos = np.array([land_x, ground_y])
        
        return gen_swing_trajectory(phase, start_pos, mid_pos, end_pos)
    
    def animate(self, frame):
        """Animation update - move forward in time."""
        if self.animation_running:
            dt = 0.033  # ~30fps
            freq = self.params['frequency']
            self.phase += dt * freq  # Move FORWARD in phase
            
        self.draw_robot()
        return []
    
    def draw_robot(self):
        """Draw the robot side view with curved leg trajectories."""
        self.ax_robot.clear()
        self.ax_robot.set_facecolor('#0d1b2a')
        
        results = self.calculate()
        land_pos = results['land_position']
        swing_ratio = self.params['swing_ratio']
        freq = self.params['frequency']
        
        # Ground
        self.ax_robot.axhline(y=0, color='#5c5c5c', linewidth=3)
        self.ax_robot.fill_between([-2.5, 2.5], [-0.03, -0.03], [0, 0], 
                                   color='#2a2a3a', alpha=0.8)
        
        # Robot body
        body_x = 0
        body_y = self.base_height
        
        body = FancyBboxPatch(
            (body_x - self.body_length/2, body_y - self.body_height/2),
            self.body_length, self.body_height,
            boxstyle="round,pad=0.02",
            facecolor='#5c7cfa',
            edgecolor='#8da4fc',
            linewidth=2
        )
        self.ax_robot.add_patch(body)
        
        # Direction arrow
        self.ax_robot.annotate('', 
            xy=(body_x + self.body_length/2 + 0.06, body_y),
            xytext=(body_x + self.body_length/2 - 0.02, body_y),
            arrowprops=dict(arrowstyle='->', color='white', lw=2))
        
        # Hip positions
        front_hip_x = body_x + self.body_length/4
        rear_hip_x = body_x - self.body_length/4
        hip_y = body_y - self.body_height/2
        
        # Draw hips
        for hx in [front_hip_x, rear_hip_x]:
            hip = Circle((hx, hip_y), 0.018, color='#ff7043', zorder=5)
            self.ax_robot.add_patch(hip)
        
        # Phase calculations
        cycle_progress = self.phase % 1.0
        stance_ratio = 1.0 - swing_ratio
        
        # Front leg phase
        front_in_swing = cycle_progress >= stance_ratio
        front_stance_progress = 0
        front_swing_phase = None
        if front_in_swing:
            front_swing_phase = (cycle_progress - stance_ratio) / swing_ratio
        else:
            front_stance_progress = cycle_progress / stance_ratio
        
        # Rear leg phase (offset by 0.5)
        rear_cycle = (cycle_progress + 0.5) % 1.0
        rear_in_swing = rear_cycle >= stance_ratio
        rear_stance_progress = 0
        rear_swing_phase = None
        if rear_in_swing:
            rear_swing_phase = (rear_cycle - stance_ratio) / swing_ratio
        else:
            rear_stance_progress = rear_cycle / stance_ratio
        
        # Trajectory endpoints
        # The foot starts swing from where it ended stance (behind the hip)
        # and lands at land_pos (ahead if accelerating needs braking, behind if needs to accelerate)
        # 
        # During stance at velocity v, the foot moves backward relative to body by: v * T_stance
        # So swing starts at approximately: -v * T_stance / 2 (middle of where it was during stance)
        # But for visualization, we simplify: start is opposite side of land
        
        # More physically accurate: start position is where foot was at end of stance
        # which is approximately: -baseline (the foot drifted back during stance)
        swing_start_offset = -results['baseline']  # Foot drifted back during stance
        
        front_start = front_hip_x + swing_start_offset
        front_land = front_hip_x + land_pos
        rear_start = rear_hip_x + swing_start_offset
        rear_land = rear_hip_x + land_pos
        
        # Draw Bezier trajectory CURVES (thick, visible)
        front_traj = self.get_bezier_trajectory(front_start, front_land)
        rear_traj = self.get_bezier_trajectory(rear_start, rear_land)
        
        # Draw curves with good visibility
        self.ax_robot.plot(front_traj[:, 0], front_traj[:, 1], 
                          '-', color='#4fc3f7', alpha=0.7, linewidth=3,
                          label='Front swing path')
        self.ax_robot.plot(rear_traj[:, 0], rear_traj[:, 1],
                          '-', color='#ffb74d', alpha=0.7, linewidth=3,
                          label='Rear swing path')
        
        # Draw legs
        def draw_leg(hip_x, swing_phase, leg_color, is_swing, start_x, land_x, stance_prog=0):
            if is_swing and swing_phase is not None:
                # Swing leg follows Bezier curve
                foot_pos = self.get_foot_position(swing_phase, start_x, land_x)
                foot_x, foot_y = foot_pos
                foot_color = '#66bb6a'  # Green for swing
            else:
                # Stance leg on ground
                # Foot starts at land_x (where it landed) and moves backward as body moves forward
                # At stance_prog=0, foot is at land position
                # At stance_prog=1, foot is at start position (ready to lift off)
                stance_travel = land_x - start_x  # How far foot travels during stance
                foot_x = land_x - stance_prog * stance_travel
                foot_y = 0
                foot_color = '#ff7043'  # Orange for stance
            
            # Draw leg as curved line (upper + lower segments)
            knee_x = (hip_x + foot_x) / 2
            knee_y = (hip_y + foot_y) / 2 - 0.02  # Slight bend
            
            # Upper leg
            self.ax_robot.plot([hip_x, knee_x], [hip_y, knee_y], 
                              color=leg_color, linewidth=5, solid_capstyle='round', zorder=3)
            # Lower leg
            self.ax_robot.plot([knee_x, foot_x], [knee_y, foot_y],
                              color=leg_color, linewidth=5, solid_capstyle='round', zorder=3)
            
            # Knee joint
            knee = Circle((knee_x, knee_y), 0.012, color='#9e9e9e', zorder=4)
            self.ax_robot.add_patch(knee)
            
            # Foot
            foot = Circle((foot_x, foot_y + 0.015), 0.022, color=foot_color, zorder=6)
            self.ax_robot.add_patch(foot)
        
        # Draw legs
        draw_leg(front_hip_x, front_swing_phase, '#4fc3f7', front_in_swing,
                front_start, front_land, front_stance_progress)
        draw_leg(rear_hip_x, rear_swing_phase, '#ffb74d', rear_in_swing,
                rear_start, rear_land, rear_stance_progress)
        
        # Landing position marker
        land_marker_x = front_hip_x + land_pos
        if land_pos >= 0:
            marker_color = '#66bb6a'
            direction = 'AHEAD'
        else:
            marker_color = '#ef5350'
            direction = 'BEHIND'
        
        self.ax_robot.axvline(x=land_marker_x, color=marker_color, 
                             linestyle='--', alpha=0.8, linewidth=2)
        self.ax_robot.plot(land_marker_x, 0, 'v', color=marker_color, markersize=15, zorder=10)
        
        # Landing position label
        self.ax_robot.text(land_marker_x, 0.42, 
                          f'LAND: {land_pos:+.3f}m\n({direction})',
                          fontsize=13, color=marker_color, fontweight='bold',
                          ha='center', va='bottom',
                          bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e', 
                                   edgecolor=marker_color, linewidth=2, alpha=0.95))
        
        # Hip reference line
        self.ax_robot.axvline(x=front_hip_x, color='#666666', linestyle=':', alpha=0.5)
        self.ax_robot.text(front_hip_x, -0.05, 'HIP', fontsize=9, color='#888888', ha='center')
        
        # Velocity arrows
        v_curr = self.params['v_current']
        v_des = v_curr + self.params['delta_v']  # Compute v_desired from delta_v
        arrow_y = body_y + 0.12
        arrow_scale = 0.025
        
        # Current velocity
        if v_curr > 0:
            self.ax_robot.annotate('', 
                xy=(body_x + 0.05 + v_curr * arrow_scale, arrow_y),
                xytext=(body_x + 0.05, arrow_y),
                arrowprops=dict(arrowstyle='->', color='#4fc3f7', lw=3))
            self.ax_robot.text(body_x + 0.05 + v_curr * arrow_scale + 0.03, arrow_y,
                              f'v = {v_curr:.1f} m/s', color='#4fc3f7', fontsize=11, 
                              fontweight='bold', va='center')
        
        # Desired velocity
        if v_des > 0:
            self.ax_robot.annotate('',
                xy=(body_x + 0.05 + v_des * arrow_scale, arrow_y - 0.05),
                xytext=(body_x + 0.05, arrow_y - 0.05),
                arrowprops=dict(arrowstyle='->', color='#ffeb3b', lw=2, linestyle='--'))
            self.ax_robot.text(body_x + 0.05 + v_des * arrow_scale + 0.03, arrow_y - 0.05,
                              f'v_des = {v_des:.1f} m/s', color='#ffeb3b', fontsize=10, 
                              va='center', alpha=0.9)
        
        # Foot height indicator
        foot_h = self.params['foot_height']
        fh_x = -0.6
        self.ax_robot.plot([fh_x, fh_x], [0, foot_h], color='#ce93d8', linewidth=2)
        self.ax_robot.plot([fh_x-0.02, fh_x+0.02], [0, 0], color='#ce93d8', linewidth=2)
        self.ax_robot.plot([fh_x-0.02, fh_x+0.02], [foot_h, foot_h], color='#ce93d8', linewidth=2)
        self.ax_robot.text(fh_x - 0.05, foot_h/2, f'{foot_h:.2f}m', 
                          color='#ce93d8', fontsize=10, va='center', ha='right', rotation=90)
        
        # Legend
        legend_elements = [
            Line2D([0], [0], color='#4fc3f7', linewidth=4, label='Front leg'),
            Line2D([0], [0], color='#ffb74d', linewidth=4, label='Rear leg'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#66bb6a', 
                   markersize=10, label='Swing foot', linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7043',
                   markersize=10, label='Stance foot', linestyle='None'),
        ]
        self.ax_robot.legend(handles=legend_elements, loc='upper right',
                            facecolor='#1a1a2e', edgecolor='#4a4a6a',
                            labelcolor='white', fontsize=10)
        
        # Axis settings - auto-scale to show landing position
        x_extent = max(0.8, abs(land_pos) + 0.4)
        self.ax_robot.set_xlim(-x_extent, x_extent)
        self.ax_robot.set_ylim(-0.08, 0.55)
        self.ax_robot.set_aspect('equal')
        self.ax_robot.set_xlabel('X Position (m)', color='white', fontsize=11)
        self.ax_robot.set_ylabel('Height (m)', color='white', fontsize=11)
        self.ax_robot.tick_params(colors='white', labelsize=10)
        self.ax_robot.grid(True, alpha=0.15, color='#666666')
        
        for spine in self.ax_robot.spines.values():
            spine.set_color('#4a4a6a')
        
        # Title with phase
        phase_pct = (self.phase % 1.0) * 100
        self.ax_robot.set_title(f'Gait Phase: {phase_pct:.0f}%', 
                               color='white', fontsize=12, pad=8)


def main():
    print("=" * 60)
    print("  Raibert Calculator - Side View")
    print("=" * 60)
    print("\n  - Sliders on the right control parameters")
    print("  - Formula and calculations shown below robot")
    print("  - Green foot = swing, Orange foot = stance")
    print("  - Curves show Bezier swing trajectory")
    print("=" * 60)
    
    calc = RaibertSideViewCalculator()
    plt.show()


if __name__ == "__main__":
    main()
