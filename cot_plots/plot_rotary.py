import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.widgets import Slider

data_dir = Path("")

files = [
    "cot_data_g0_rotary.csv",
    "cot_data_g2_rotary.csv",
    "cot_data_ge_rotary.csv",
    "cot_data_gg_rotary.csv"
]

# Load all data once
datasets = []
for file in files:
    df = pd.read_csv(data_dir / file)
    datasets.append((file.replace(".csv", ""), df))

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.2)

scatter_plots = []
line_plots = []

# Initial frac value
init_frac = 0.25

# Plot initial data
for label, df in datasets:
    sc = ax.scatter(
        df["velocity"],
        df["cot"],
        s=20,
        alpha=0.3
    )

    smoothed = lowess(df["cot"], df["velocity"], frac=init_frac)
    ln, = ax.plot(
        smoothed[:, 0],
        smoothed[:, 1],
        linewidth=2.5,
        label=label
    )

    scatter_plots.append(sc)
    line_plots.append(ln)

ax.set_xlabel("Velocity")
ax.set_ylabel("COT")
ax.set_title("COT vs Velocity with Interactive LOWESS")
ax.legend()
ax.grid(True)

# Slider axis
ax_frac = plt.axes([0.2, 0.05, 0.6, 0.03])
frac_slider = Slider(
    ax=ax_frac,
    label="LOWESS frac",
    valmin=0.05,
    valmax=0.9,
    valinit=init_frac,
    valstep=0.01
)

# Update function
def update(val):
    frac = frac_slider.val
    for (label, df), line in zip(datasets, line_plots):
        smoothed = lowess(df["cot"], df["velocity"], frac=frac)
        line.set_data(smoothed[:, 0], smoothed[:, 1])
    fig.canvas.draw_idle()

frac_slider.on_changed(update)

plt.show()

