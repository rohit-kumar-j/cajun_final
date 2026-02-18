"""
Interactive COT vs Velocity plot for multiple gaits.
Hover over legend entries to highlight that gait's data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path
import os


def load_gait_data(data_dir, data_range=(0, None)):
    """Load data from a gait directory and calculate COT per stride."""
    data_dir = Path(data_dir)
    start_idx, end_idx = data_range
    
    # Load required files
    try:
        torso_x = np.loadtxt(data_dir / 'PosTorso0.txt')
        front_stance = np.loadtxt(data_dir / 'desPosTorso9.txt')
        time_data = np.loadtxt(data_dir / 'time.txt')
        
        # Apply range clipping
        if end_idx is None:
            end_idx = len(torso_x)
        end_idx = min(end_idx, len(torso_x))
        start_idx = max(0, start_idx)
        
        torso_x = torso_x[start_idx:end_idx]
        front_stance = front_stance[start_idx:end_idx]
        time_data = time_data[start_idx:end_idx]
        
        # Load joint data with same range
        tau = np.zeros((12, len(torso_x)))
        dq = np.zeros((12, len(torso_x)))
        for i in range(12):
            tau_full = np.loadtxt(data_dir / f'tauM{i}.txt')
            dq_full = np.loadtxt(data_dir / f'dq{i}.txt')
            tau[i] = tau_full[start_idx:end_idx]
            dq[i] = dq_full[start_idx:end_idx]
    except Exception as e:
        print(f"Error loading {data_dir}: {e}")
        return None, None
    
    # Parameters (A1 robot)
    mass = 4.713 + 0.001 + 0.696*4 + 1.013*4 + 0.166*4 + 0.06*4  # ~12.5 kg
    g = 9.81
    dt = 0.002
    
    # Detect stride boundaries using front stance transitions
    # Front stance: 5 = stance, 0 = swing
    front_binary = (front_stance == 5).astype(int)
    front_diff = np.diff(front_binary)
    
    # Find liftoff (stance -> swing = -1) indices
    liftoffs = np.where(front_diff == -1)[0]
    
    if len(liftoffs) < 2:
        print(f"Not enough strides in {data_dir}")
        return None, None
    
    # Calculate COT for each stride
    velocities = []
    cots = []
    
    for i in range(len(liftoffs) - 1):
        start_idx = liftoffs[i] + 1
        end_idx = liftoffs[i + 1]
        
        if end_idx <= start_idx:
            continue
        
        # Distance traveled
        x_i = torso_x[start_idx]
        x_f = torso_x[end_idx]
        distance = x_f - x_i
        
        if distance <= 0:
            continue
        
        # Time for stride
        t_i = time_data[start_idx]
        t_f = time_data[end_idx]
        stride_time = t_f - t_i
        
        if stride_time <= 0:
            continue
        
        # Velocity
        velocity = distance / stride_time
        
        # Mechanical work (absolute value of power integrated)
        power = tau[:, start_idx:end_idx] * dq[:, start_idx:end_idx]
        work = np.sum(np.abs(power)) * dt
        
        # Gravitational work (normalization)
        grav_work = mass * g * abs(distance)
        
        # COT
        cot = work / grav_work
        
        # Filter reasonable values
        if 0 < velocity < 10 and 0 < cot < 20:
            velocities.append(velocity)
            cots.append(cot)
    
    return np.array(velocities), np.array(cots)


class InteractiveCOTPlot:
    """Interactive COT vs Velocity plot with hover-to-highlight legend."""
    
    def __init__(self, base_path='logs_raibert/cot_data', variants=None, data_range=(0, None)):
        self.base_path = Path(base_path)
        self.variants = variants if variants else ['transverse', 'rotary']
        self.data_range = data_range  # (start, end) indices
        
        # Create figure based on how many variants
        if len(self.variants) == 2:
            self.fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            self.axes = {'transverse': axes[0], 'rotary': axes[1]}
        else:
            self.fig, ax = plt.subplots(figsize=(7, 5))
            self.axes = {self.variants[0]: ax}
        
        self.fig.tight_layout(pad=2.5)
        
        # Colors for different gait classes
        self.colors = {
            'g0': '#1f77b4',    # blue
            'g2': '#d62728',    # red
            'gg': '#2ca02c',    # green
            'ge': '#9467bd',    # purple
        }
        
        # Markers for gait classes
        self.markers = {
            'g0': 'o',
            'g2': 's',
            'gg': '^',
            'ge': 'D',
        }
        
        self.scatter_plots = {v: {} for v in self.variants}
        self.legend_items = {v: {} for v in self.variants}
        self.data = {}
        
        self.load_all_data()
        self.setup_plot()
        self.setup_legend()
        self.connect_events()
    
    def load_all_data(self):
        """Load data from all gait directories."""
        gait_dirs = sorted(self.base_path.glob('*_data'))
        
        for gait_dir in gait_dirs:
            gait_name = gait_dir.name.replace('_data', '')
            print(f"Loading {gait_name}...")
            
            vel, cot = load_gait_data(gait_dir, data_range=self.data_range)
            if vel is not None and len(vel) > 0:
                self.data[gait_name] = {'velocity': vel, 'cot': cot}
                print(f"  Loaded {len(vel)} strides")
            else:
                print(f"  No valid data")
    
    def setup_plot(self):
        """Create scatter plots for each gait on appropriate subplot."""
        
        # Setup axes
        titles = {'transverse': 'Transverse Gaits', 'rotary': 'Rotary Gaits'}
        for variant in self.variants:
            ax = self.axes[variant]
            ax.set_xlabel('Velocity (m/s)', fontsize=11)
            ax.set_ylabel('Cost of Transport (COT)', fontsize=11)
            ax.set_title(titles[variant], fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 1.0)
        
        for gait_name, gait_data in self.data.items():
            # Determine gait class (g0, g2, gg, ge) and variant (transverse/rotary)
            parts = gait_name.split('_')
            gait_class = parts[0]
            variant = parts[1] if len(parts) > 1 else 'transverse'
            
            # Skip if this variant isn't being shown
            if variant not in self.variants:
                continue
            
            color = self.colors.get(gait_class, '#333333')
            marker = self.markers.get(gait_class, 'o')
            
            ax = self.axes[variant]
            
            scatter = ax.scatter(
                gait_data['velocity'],
                gait_data['cot'],
                c=color,
                marker=marker,
                s=50,
                alpha=0.7,
                label=gait_class.upper(),
                picker=True,
                pickradius=5
            )
            self.scatter_plots[variant][gait_class] = scatter
    
    def setup_legend(self):
        """Create interactive legend for each subplot."""
        self.legends = {}
        
        for variant in self.variants:
            ax = self.axes[variant]
            
            # Remove duplicate labels
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            
            legend = ax.legend(
                by_label.values(), 
                by_label.keys(),
                loc='upper right',
                fontsize=10,
                framealpha=0.9,
                markerscale=1.2
            )
            
            self.legends[variant] = legend
            
            # Store legend artists for interaction
            try:
                handles = legend.legend_handles
            except AttributeError:
                handles = legend.legendHandles
            
            for legend_item, gait_class in zip(handles, by_label.keys()):
                self.legend_items[variant][legend_item] = gait_class.lower()
                legend_item.set_picker(True)
                legend_item.set_pickradius(10)
    
    def connect_events(self):
        """Connect mouse events for interactivity."""
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        self.highlighted = {v: None for v in self.variants}
    
    def highlight_gait(self, variant, gait_class):
        """Highlight a specific gait in a subplot, dim others."""
        if self.highlighted[variant] == gait_class:
            return
        
        self.highlighted[variant] = gait_class
        
        for name, scatter in self.scatter_plots[variant].items():
            if name == gait_class:
                scatter.set_alpha(1.0)
                scatter.set_sizes([120])  # Larger
                scatter.set_zorder(10)
            else:
                scatter.set_alpha(0.15)
                scatter.set_sizes([30])  # Smaller
                scatter.set_zorder(1)
        
        # Update legend
        for legend_item, name in self.legend_items[variant].items():
            if name == gait_class:
                legend_item.set_alpha(1.0)
            else:
                legend_item.set_alpha(0.3)
        
        self.fig.canvas.draw_idle()
    
    def reset_highlight(self, variant):
        """Reset all gaits to normal appearance in a subplot."""
        if self.highlighted[variant] is None:
            return
        
        self.highlighted[variant] = None
        
        for scatter in self.scatter_plots[variant].values():
            scatter.set_alpha(0.7)
            scatter.set_sizes([50])
            scatter.set_zorder(2)
        
        for legend_item in self.legend_items[variant].keys():
            legend_item.set_alpha(1.0)
        
        self.fig.canvas.draw_idle()
    
    def get_variant_from_axes(self, ax):
        """Determine which variant (transverse/rotary) based on axes."""
        for variant, axis in self.axes.items():
            if ax == axis:
                return variant
        return None
    
    def on_hover(self, event):
        """Handle mouse hover events."""
        if event.inaxes is None:
            return
        
        variant = self.get_variant_from_axes(event.inaxes)
        if variant is None:
            return
        
        # Check if hovering over data points
        found = False
        for gait_class, scatter in self.scatter_plots[variant].items():
            cont, ind = scatter.contains(event)
            if cont:
                self.highlight_gait(variant, gait_class)
                found = True
                break
        
        if not found and self.highlighted[variant]:
            self.reset_highlight(variant)
    
    def on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes is None:
            return
        
        variant = self.get_variant_from_axes(event.inaxes)
        if variant is None:
            return
        
        # Check if clicking on data points
        for gait_class, scatter in self.scatter_plots[variant].items():
            cont, ind = scatter.contains(event)
            if cont:
                if self.highlighted[variant] == gait_class:
                    self.reset_highlight(variant)
                else:
                    self.highlight_gait(variant, gait_class)
                return
    
    def show(self):
        """Display the plot."""
        plt.show()


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Interactive COT vs Velocity Plot')
    parser.add_argument('--path', default='.', 
                        help='Base path containing gait data directories (default: current directory)')
    parser.add_argument('--transverse', action='store_true',
                        help='Show only transverse gaits')
    parser.add_argument('--rotary', action='store_true',
                        help='Show only rotary gaits')
    parser.add_argument('--start', type=int, default=0,
                        help='Start index for data range (default: 0)')
    parser.add_argument('--end', type=int, default=None,
                        help='End index for data range (default: all data)')
    args = parser.parse_args()
    
    # Determine which variants to show
    if args.transverse and args.rotary:
        variants = ['transverse', 'rotary']  # Both
    elif args.transverse:
        variants = ['transverse']
    elif args.rotary:
        variants = ['rotary']
    else:
        variants = ['transverse', 'rotary']  # Default: both
    
    # Data range
    data_range = (args.start, args.end)
    
    # The data directories are named like g0_rotary_data, g0_transverse_data, etc.
    # They live directly inside logs_raibert/cot_data/
    if not Path(args.path).exists():
        print(f"Path not found: {args.path}")
        print("Expected structure:")
        print("  logs_raibert/cot_data/")
        print("    g0_transverse_data/")
        print("    g0_rotary_data/")
        print("    g2_transverse_data/")
        print("    ...")
        return
    
    print("Creating interactive COT plot...")
    print("Hover over data points to highlight that gait")
    print("Click to toggle highlight lock")
    if args.start > 0 or args.end is not None:
        print(f"Data range: {args.start} to {args.end if args.end else 'end'}")
    
    plot = InteractiveCOTPlot(args.path, variants=variants, data_range=data_range)
    plot.show()


if __name__ == '__main__':
    main()



