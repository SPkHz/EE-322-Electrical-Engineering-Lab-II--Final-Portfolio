import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file for ID vs. VDS family of curves
file_path_vds = 'NMOS_Parametric_Sweep.csv'
data_vds = pd.read_csv(file_path_vds)

# Load the CSV file for ID vs. VGS family of curves
file_path_vgs = 'VGS_NMOS_Parametric_Sweep.csv'
data_vgs = pd.read_csv(file_path_vgs)

# Prepare dictionaries to store data_vds by VGS and data_vgs by VDS
all_data_vds = {}
all_data_vgs = {}

for col in data_vds.columns:
    if 'VGS=' in col and '.1' not in col:
        vgs_label = float(col.split('=')[1].replace('V', ''))  # Extract numerical VGS value
        vds = pd.to_numeric(data_vds[col], errors='coerce').dropna().values
        current_vds = pd.to_numeric(data_vds[f"{col}.1"], errors='coerce').dropna().values
        all_data_vds[vgs_label] = {'VDS': vds, 'Current': current_vds}

for col in data_vgs.columns:
    if 'VDS=' in col and '.1' not in col:
        vds_label = float(col.split('=')[1].replace('V', ''))  # Extract numerical VDS value
        vgs = pd.to_numeric(data_vgs[col], errors='coerce').dropna().values
        current_vgs = pd.to_numeric(data_vgs[f"{col}.1"], errors='coerce').dropna().values
        all_data_vgs[vds_label] = {'VGS': vgs, 'Current': current_vgs}

# Create subplots (2 rows, 2 columns)
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('NMOS Characterization and Analysis', fontsize=18)

# Plot ID vs. VDS data (top left)
for vgs_label, data_vds in all_data_vds.items():
    axs[0, 0].scatter(data_vds['VDS'], data_vds['Current'] * 1e3, label=f"$V_{{GS}}$ = {vgs_label:.1f}")
axs[0, 0].set_title('$I_{D}$ vs. $V_{DS}$ Curves')
axs[0, 0].set_xlabel('$V_{DS}$ (V)')
axs[0, 0].set_ylabel('$I_{D}$ (mA)')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot ID vs. VGS data (top right)
for vds_label, data_vgs in all_data_vgs.items():
    axs[0, 1].scatter(data_vgs['VGS'], data_vgs['Current'] * 1e3, label=f"$V_{{DS}}$ = {vds_label:.1f}")
axs[0, 1].set_title('$I_{D}$ vs. $V_{GS}$ Curves')
axs[0, 1].set_xlabel('$V_{GS}$ (V)')
axs[0, 1].set_ylabel('$I_{D}$ (mA)')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Analyze sqrt(ID) vs. VGS (bottom right)
kn_values = []
vth_values = []
r2_values = []

for vds_label, data_vgs in all_data_vgs.items():
    sqrt_id = np.sqrt(np.maximum(data_vgs['Current'], 0))
    mask = (data_vgs['VGS'] >= 2.0) & (data_vgs['VGS'] <= 3.0)  # Linear region



    # Perform linear fit
    fit_params = np.polyfit(data_vgs['VGS'][mask], sqrt_id[mask], 1)
    kn = (fit_params[0] ** 2) * 1e6  # µA/V²
    vth = -fit_params[1] / fit_params[0]
    kn_values.append(kn)
    vth_values.append(vth)

    # Calculate R^2 value
    y_pred = np.polyval(fit_params, data_vgs['VGS'][mask])
    r2 = 1 - np.sum((sqrt_id[mask] - y_pred) ** 2) / np.sum((sqrt_id[mask] - np.mean(sqrt_id[mask])) ** 2)
    r2_values.append(r2)

    # Plot data points
    axs[1, 1].scatter(data_vgs['VGS'][mask], sqrt_id[mask], label=f"$V_{{DS}}$ = {vds_label:.1f}", alpha=0.7)

    # Plot linear fit
    axs[1, 1].plot(
        data_vgs['VGS'][mask],
        y_pred,
        linestyle='--',
        label=f"Fit $V_{{DS}}$ = {vds_label:.1f}"
    )

# Calculate mean and standard deviation of parameters
kn_mean, kn_std = np.mean(kn_values), np.std(kn_values)
vth_mean, vth_std = np.mean(vth_values), np.std(vth_values)
r2_mean = np.mean(r2_values)

# Annotate the subplot with results
axs[1, 1].text(
    0.5, 0.1,
    f"$K_N$: {kn_mean:.2f} ± {kn_std:.2f} µA/V²\n"
    f"$V_{{TH}}$: {vth_mean:.3f} ± {vth_std:.3f} V\n"
    f"$R^{{2}}$: {r2_mean:.4f}",
    transform=axs[1, 1].transAxes,
    fontsize=12,
    ha='center',
    bbox=dict(facecolor='white', edgecolor='black')
)

# Finalize the plot
axs[1, 1].set_title(r'$\sqrt{I_{D}}$ vs. $V_{GS}$ with Linear Fits')
axs[1, 1].set_xlabel('$V_{GS}$ (V)')
axs[1, 1].set_ylabel(r'$\sqrt{I_{D}}$ (A$^{1/2}$)')
axs[1, 1].legend(loc='upper left')
axs[1, 1].grid(True)

# Analyze ID vs. VDS for VA (bottom left)
va_values = []

# Plot ID vs VDS and the linear fits for V_A calculation
for vgs_label, data_vds in all_data_vds.items():
    # Filter the data based on the linear range: VDS >= VGS - VTH_mean and VDS <= 8V
    linear_mask = (data_vds['VDS'] >= vgs_label - vth_mean) & (data_vds['VDS'] <= 8.0)
    vds_linear = data_vds['VDS'][linear_mask]
    id_linear = data_vds['Current'][linear_mask] * 1e3  # Scale current to mA

    # Perform linear fit
    fit_params = np.polyfit(vds_linear, id_linear, 1)  # Linear fit
    va = -fit_params[1] / fit_params[0]  # X-intercept (V_A)
    va_values.append(va)

    # Plot the data points and linear fit
    axs[1, 0].scatter(vds_linear, id_linear, label=f"$V_{{GS}}$ = {vgs_label:.1f}", s=20)
    axs[1, 0].plot(
        vds_linear,
        np.polyval(fit_params, vds_linear),
        linestyle="--",
        label=f"Fit ($V_{{GS}}$ = {vgs_label:.1f})",
    )

# Format the bottom-left subplot
axs[1, 0].set_title("$I_{D}$ vs. $V_{DS}$ with Linear Fits")
axs[1, 0].set_xlabel("$V_{DS}$ (V)")
axs[1, 0].set_ylabel("$I_{D}$ (mA)")
axs[1, 0].legend(fontsize=8)
axs[1, 0].grid(True)
# Annotate the subplot with individual VA results
va_text = "\n".join([f"$V_{{GS}}$={vgs_label:.1f}: $V_A$ = {va:.2f} V" for vgs_label, va in zip(all_data_vds.keys(), va_values)])
axs[1, 0].text(
    0.5, 0.2,
    va_text,
    transform=axs[1, 0].transAxes,
    fontsize=10,
    ha='center',
    bbox=dict(facecolor='white', edgecolor='black')
)


axs[1, 0].set_title('$V_A$ Analysis')
axs[1, 0].set_xlabel('$V_{DS}$ (V)')
axs[1, 0].set_ylabel('$I_{D}$ (mA)')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Adjust layout and save
plt.tight_layout()
plt.savefig('NMOS_Characterization_Analysis.png', dpi=300)
plt.show()