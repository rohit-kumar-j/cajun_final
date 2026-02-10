"""
COT vs Velocity plot for gaits.
Can be called standalone or as subprocess from evalcot.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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


def create_cot_plot(base_path, data_range=(0, None), save_path=None):
    """
    Create COT vs Velocity plot.
    
    Args:
        base_path: Directory containing gait data subdirectories or a single gait directory
        data_range: (start_idx, end_idx) for data slicing
        save_path: If provided, save plot to this path instead of showing
    """
    base_path = Path(base_path)
    
    # Check if this is a single gait directory or contains multiple
    has_data_files = (base_path / 'PosTorso0.txt').exists()
    
    if has_data_files:
        # Single gait directory - plot just this one
        print(f"Loading single gait from {base_path}...")
        vel, cot = load_gait_data(base_path, data_range=data_range)
        
        if vel is None or len(vel) == 0:
            print("No valid data found")
            return None
        
        # Create simple scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(vel, cot, c='#1f77b4', marker='o', s=50, alpha=0.7)
        ax.set_xlabel('Velocity (m/s)', fontsize=12)
        ax.set_ylabel('Cost of Transport (COT)', fontsize=12)
        ax.set_title('Cost of Transport vs Velocity', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(vel) * 1.1)
        ax.set_ylim(0, max(cot) * 1.1)
        
        print(f"Loaded {len(vel)} strides")
        
    else:
        # Multiple gait directories
        gait_dirs = sorted(base_path.glob('*_data'))
        
        if len(gait_dirs) == 0:
            print(f"No gait data directories found in {base_path}")
            return None
        
        data = {}
        for gait_dir in gait_dirs:
            gait_name = gait_dir.name.replace('_data', '')
            print(f"Loading {gait_name}...")
            
            vel, cot = load_gait_data(gait_dir, data_range=data_range)
            if vel is not None and len(vel) > 0:
                data[gait_name] = {'velocity': vel, 'cot': cot}
                print(f"  Loaded {len(vel)} strides")
        
        if len(data) == 0:
            print("No valid data found")
            return None
        
        # Create plot with all gaits
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = {
            'g0': '#1f77b4',    # blue
            'g2': '#d62728',    # red
            'gg': '#2ca02c',    # green
            'ge': '#9467bd',    # purple
        }
        
        markers = {
            'g0': 'o',
            'g2': 's',
            'gg': '^',
            'ge': 'D',
        }
        
        for gait_name, gait_data in data.items():
            # Extract gait class (g0, g2, etc)
            parts = gait_name.split('_')
            gait_class = parts[0]
            
            color = colors.get(gait_class, '#333333')
            marker = markers.get(gait_class, 'o')
            
            ax.scatter(
                gait_data['velocity'],
                gait_data['cot'],
                c=color,
                marker=marker,
                s=50,
                alpha=0.7,
                label=gait_class.upper()
            )
        
        ax.set_xlabel('Velocity (m/s)', fontsize=12)
        ax.set_ylabel('Cost of Transport (COT)', fontsize=12)
        ax.set_title('Cost of Transport vs Velocity', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 1.0)
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()
    
    return fig


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='COT vs Velocity Plot')
    parser.add_argument('--path', required=True,
                        help='Path to gait data directory')
    parser.add_argument('--start', type=int, default=0,
                        help='Start index for data range (default: 0)')
    parser.add_argument('--end', type=int, default=None,
                        help='End index for data range (default: all data)')
    parser.add_argument('--save', action='store_true',
                        help='Save plot instead of showing interactively')
    args = parser.parse_args()
    
    data_range = (args.start, args.end)
    
    if args.start > 0 or args.end is not None:
        print(f"Data range: {args.start} to {args.end if args.end else 'end'}")
    
    # Determine save path
    save_path = None
    if args.save:
        save_path = Path(args.path) / 'cot_plot.png'
    
    create_cot_plot(args.path, data_range=data_range, save_path=save_path)


if __name__ == '__main__':
    main()
