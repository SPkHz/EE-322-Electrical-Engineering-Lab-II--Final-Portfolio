import ltspice
import numpy as np
import matplotlib.pyplot as plt

# Function to plot data from a given .raw file
def plot_v_output(filename, label, color):
    l = ltspice.Ltspice(filename)
    l.parse()

    time = l.get_data('time')
    V_vo = l.get_data('V(vo)')

    plt.plot(time, V_vo, label=label, color=color)
    plt.title(label)
    plt.xlabel("Time (s)")
    plt.ylabel("V(vo) (V)")
    plt.grid(True)
    plt.legend(loc='upper right')

    # Add y-axis margin
    margin = 0.085
    y_min, y_max = min(V_vo), max(V_vo)
    y_range = y_max - y_min
    plt.ylim(y_min - margin * y_range, y_max + margin * y_range)

    # --- Max Value Annotation ---
    idx_max = np.argmax(V_vo)
    t_max, v_max = time[idx_max], V_vo[idx_max]
    plt.plot(t_max, v_max, 'go')  # green dot
    plt.text(
        t_max, v_max + y_range * 0.035,
        f"Max: {v_max:.3f} V",
        color='black',
        ha='center',
        bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.3')
    )

    # --- Min Value Annotation ---
    idx_min = np.argmin(V_vo)
    t_min, v_min = time[idx_min], V_vo[idx_min]
    plt.plot(t_min, v_min, 'ro')  # red dot
    plt.text(
        t_min, v_min - y_range * 0.055,
        f"Min: {v_min:.3f} V",
        color='black',
        ha='center',
        bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.3')
    )

    plt.tight_layout()
    plt.show()


# File names
filename_180_deg = "EE322_Lab_07_Common_Mode_Placzek.raw"
filename_common_mode = "EE322_Lab_07_Differential_Input_Placzek.raw"

# Create and show two separate plots
plt.figure(figsize=(10, 6))
plot_v_output(filename_180_deg, label="AC - Differential Input V(vo)", color='blue')

plt.figure(figsize=(10, 6))
plot_v_output(filename_common_mode, label="AC - Common Mode V(vo)", color='red')