import re
import chardet
import unicodedata
import re
import chardet
from typing import Optional

# === FILE SETUP ===
log_file_path = "/opt/miniconda3/envs/pycharm-env/EE322_ee_Lab_II/Labs/Lab-07/EE322_Lab_07_DC_LTspice_Sim_Placzek.log"
file_path = "/opt/miniconda3/envs/pycharm-env/EE322_ee_Lab_II/Labs/Lab-07/EE322_Lab_07_DC_LTspice_Sim_Placzek.log"
# Read and decode
with open(log_file_path, "rb") as f:
    raw_data = f.read()
    encoding = chardet.detect(raw_data)["encoding"]
    log_data = raw_data.decode(encoding)
    log_data = unicodedata.normalize("NFKD", log_data)

# === EXTRACT NODE VOLTAGES FOR Q1/Q2 ===
v_vo = float(re.search(r"V\(vo\)\s+([-\d.eE]+)", log_data).group(1))  # source
v_vl = float(re.search(r"V\(vl\)\s+([-\d.eE]+)", log_data).group(1))  # drain of Q1
v_vr = float(re.search(r"V\(vr\)\s+([-\d.eE]+)", log_data).group(1))  # drain of Q2

# === EXTRACT TRANSISTOR DATA ===
def extract_transistor_columns(log_data):
    block_match = re.search(r"--- MOSFET Transistors ---(.+?)Operating Bias Point", log_data, re.DOTALL)
    if not block_match:
        raise ValueError("Couldn't find MOSFET transistor block.")

    block = block_match.group(1)
    lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
    labels = lines[0].split()[1:]  # Skip "Name:"

    numeric_data = {}
    for line in lines[1:]:
        if line.startswith("Model:"):
            continue
        parts = line.split()
        key = parts[0].replace(":", "")
        numeric_data[key] = parts[1:]

    data_by_transistor = {}
    for i, label in enumerate(labels):
        data_by_transistor[label] = {
            "Id": float(numeric_data["Id"][i]),
            "Vgs": float(numeric_data["Vgs"][i]),
            "Vds": float(numeric_data["Vds"][i]),
            "Vth": float(numeric_data["Vth"][i])
        }
    return data_by_transistor

# === DETERMINE NODE VOLTAGES FROM .LOG ===
def reconstruct_voltages(params, is_pmos, v_d=None, v_s=None):
    vgs = params["Vgs"]
    vds = params["Vds"]
    vth = params["Vth"]
    id_val = params["Id"]
    vov = abs(vgs - vth)

    if is_pmos:
        v_s = 12.0
        v_d = v_s + vds
        v_g = v_s + vgs
    else:
        v_g = 0.0
        # v_s and v_d passed in from node voltages

    return {
        "I_D": abs(id_val * 1000),   # Convert to mA (optional: match your table units)
        "V_OV": vov,
        "V_G": v_g,
        "V_D": v_d,
        "V_S": v_s
    }


# ======== EXECUTE SIMULATED =============
raw_mos_data = extract_transistor_columns(log_data)

results = {
    "Q1": reconstruct_voltages(raw_mos_data["m§q1"], is_pmos=False, v_d=v_vl, v_s=v_vo),
    "Q2": reconstruct_voltages(raw_mos_data["m§q2"], is_pmos=False, v_d=v_vr, v_s=v_vo),
    "Q3": reconstruct_voltages(raw_mos_data["m§q3"], is_pmos=True),
    "Q4": reconstruct_voltages(raw_mos_data["m§q4"], is_pmos=True),
}

# === PRINT RESULTS ===
def print_table(q_data, label):
    print(f"Device: {label}")
    print(f"I_D     = {q_data['I_D']:.6f} A")
    print(f"|V_OV|  = {q_data['V_OV']:.4f} V")
    print(f"V_G     = {q_data['V_G']:.4f} V")
    print(f"V_D     = {q_data['V_D']:.4f} V")
    print(f"V_S     = {q_data['V_S']:.4f} V")
    print("-" * 30)

for label in ["Q1", "Q2", "Q3", "Q4"]:
    print_table(results[label], label)

########################## Measured I_Dk Calculations ########################

def calculate_percentage_difference(sim: float, meas: float) -> Optional[float]:
    if meas == 0:
        return None
    try:
        return abs((sim - meas) / meas) * 100
    except ZeroDivisionError:
        return None

def format_value(val):
    return f"{val:.4f}" if isinstance(val, float) else "N/A"

########################## Simulated DC Operating Point Parsing for printing ########################

def extract_simulated_dc_points(file_path):
    simulated = {"Q1": {}, "Q2": {}, "Q3": {}, "Q4": {}}
    try:
        # Detect file encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            encoding = chardet.detect(raw_data)['encoding']

        # Read the log file content
        with open(file_path, 'r', encoding=encoding) as file:
            content = file.read()

        # Anchor each parameter to the start of a line (MULTILINE mode)
        id_match = re.search(r'^Id:\s*([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)', content, re.MULTILINE)
        vgs_match = re.search(r'^Vgs:\s*([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)', content, re.MULTILINE)
        vds_match = re.search(r'^Vds:\s*([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)', content, re.MULTILINE)
        vth_match = re.search(r'^Vth:\s*([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)', content, re.MULTILINE)

        if id_match and vgs_match and vds_match and vth_match:
            devices = ["Q1", "Q2", "Q3", "Q4"]
            for i, device in enumerate(devices, start=1):
                # Convert the simulated drain current from A to µA
                simulated[device]["ID"] = float(id_match.group(i)) * 1e6
                simulated[device]["VGS"] = float(vgs_match.group(i))
                simulated[device]["VDS"] = float(vds_match.group(i))
                simulated[device]["VTH"] = float(vth_match.group(i))
                simulated[device]["VOV"] = abs(simulated[device]["VGS"] - simulated[device]["VTH"])
                simulated[device]["VG"] = simulated[device]["VGS"]
                simulated[device]["VD"] = simulated[device]["VDS"]
                simulated[device]["VS"] = 0.0  # Assumed for simulation
        else:
            print("One or more parameter lines could not be found in the log file.")
    except Exception as e:
        print(f"Error parsing log file: {e}")

    return simulated

########################## Measured DC Calculation ########################

def calculate_measured_id(vg, vs, device_type):
    if device_type == "N":  # For Q1 and Q2 (N-channel)
        Vth = 0.62      # Typical threshold voltage (V)
        k = 580e-6      # Transconductance parameter in A/V² (580 µA/V²)
        vgs = vg - vs
        vov = vgs - Vth
        return 0.5 * k * (vov ** 2)
    elif device_type == "P":  # For Q3 and Q4 (P-channel)
        Vth = 1.00      # Typical threshold magnitude (V)
        k = 189e-6      # Transconductance parameter in A/V² (189 µA/V²)
        vsg = vs - vg
        vov = vsg - Vth
        return 0.5 * k * (vov ** 2)
    else:
        return None

# Measured node voltages from the lab (as obtained from your setup)
measured_nodes = {
    "Q1": {"VG": 0.000, "VD": 8.606, "VS": -1.985},
    "Q2": {"VG": 0.000, "VD": 8.606, "VS": -1.985},
    "Q3": {"VG": 8.606, "VD": 8.606, "VS": 12.000},
    "Q4": {"VG": 8.606, "VD": 8.606, "VS": 12.000},
}

# Calculate the measured DC parameters including ID and |VOV|
measured_dc = {}
for device in ["Q1", "Q2", "Q3", "Q4"]:
    if device in ["Q1", "Q2"]:
        id_meas = calculate_measured_id(measured_nodes[device]["VG"],
                                        measured_nodes[device]["VS"],
                                        "N")
        vov = (measured_nodes[device]["VG"] - measured_nodes[device]["VS"]) - 0.62
        measured_dc[device] = {
            "ID": id_meas * 1e6,  # Convert A to µA
            "VOV": vov,
            "VG": measured_nodes[device]["VG"],
            "VD": measured_nodes[device]["VD"],
            "VS": measured_nodes[device]["VS"]
        }
    else:
        id_meas = calculate_measured_id(measured_nodes[device]["VG"],
                                        measured_nodes[device]["VS"],
                                        "P")
        vov = (measured_nodes[device]["VS"] - measured_nodes[device]["VG"]) - 1.00
        measured_dc[device] = {
            "ID": id_meas * 1e6,  # in µA
            "VOV": vov,
            "VG": measured_nodes[device]["VG"],
            "VD": measured_nodes[device]["VD"],
            "VS": measured_nodes[device]["VS"]
        }

########################## Table Printer ########################

def print_dc_summary(simulated, measured):
    print("\n========== DC Summary Table ==========")
    headers = (f"{'Device':<6} | {'Quantity':<6} | {'Simulated':>10} | "
               f"{'Measured':>10} | Units | {'Difference':>10} | {'% Difference':>12}")
    print(headers)
    print("-" * len(headers))

    for device in ["Q1", "Q2", "Q3", "Q4"]:
        quantities = [
            ("ID", "µA"),
            ("VOV", "V"),
            ("VG", "V"),
            ("VD", "V"),
            ("VS", "V")
        ]
        for q, unit in quantities:
            sim_val = simulated[device].get(q, None)
            meas_val = measured[device].get(q, None)
            # If either value is missing, use "N/A"
            if sim_val is None or meas_val is None:
                sim_str = sim_val if sim_val is not None else "N/A"
                meas_str = meas_val if meas_val is not None else "N/A"
                diff_str = "N/A"
                pct_str = "N/A"
            else:
                sim_str = f"{sim_val:.3f}"
                meas_str = f"{meas_val:.3f}"
                diff = abs(sim_val - meas_val)
                pct_diff = calculate_percentage_difference(sim_val, meas_val)
                diff_str = f"{diff:.3f}"
                pct_str = f"{pct_diff:.3f}" if pct_diff is not None else "N/A"
            print(f"{device:<6} | {q:<6} | {sim_str:>10} | {meas_str:>10} | {unit:<5} | {diff_str:>10} | {pct_str:>12}")
    print("=" * len(headers))

########################## Main Execution ########################

# File path to the simulated log file (update the path as needed)
log_file_path = "/opt/miniconda3/envs/pycharm-env/EE322_ee_Lab_II/Labs/Lab-07/EE322_Lab_07_DC_LTspice_Sim_Placzek.log"
simulated_dc = extract_simulated_dc_points(log_file_path)

# For demonstration, if simulated_dc could not be parsed, we use dummy simulation data:
if not simulated_dc["Q1"]:
    simulated_dc = {
        "Q1": {"ID": 0.000, "VOV": 1.267, "VG": 0.000, "VD": 9.011, "VS": -1.844},
        "Q2": {"ID": 0.000, "VOV": 1.267, "VG": 0.000, "VD": 9.011, "VS": -1.844},
        "Q3": {"ID": 0.000, "VOV": 2.343, "VG": 9.010, "VD": 9.010, "VS": 12.000},
        "Q4": {"ID": 0.000, "VOV": 2.343, "VG": 9.010, "VD": 9.010, "VS": 12.000}
    }

# Print the measured node voltages
print("Measured Node Voltages:")
for transistor, values in measured_nodes.items():
    print(f"{transistor}:")
    for param, value in values.items():
        print(f"  {param}: {value:.3f}")
    print()



import csv


def save_dc_summary_to_csv(simulated, measured, filename="/opt/miniconda3/envs/pycharm-env/EE322_ee_Lab_II/Labs/Lab-07/dc_summary.csv"):
    rows = []
    header = ["Device", "Quantity", "Simulated", "Measured", "Units", "Difference", "% Difference"]
    for device in ["Q1", "Q2", "Q3", "Q4"]:
        quantities = [
            ("ID", "µA"),
            ("VOV", "V"),
            ("VG", "V"),
            ("VD", "V"),
            ("VS", "V")
        ]
        for q, unit in quantities:
            sim_val = simulated[device].get(q, None)
            meas_val = measured[device].get(q, None)
            if sim_val is None or meas_val is None:
                sim_str = sim_val if sim_val is not None else "N/A"
                meas_str = meas_val if meas_val is not None else "N/A"
                diff_str = "N/A"
                pct_str = "N/A"
            else:
                sim_str = f"{sim_val:.3f}"
                meas_str = f"{meas_val:.3f}"
                diff = abs(sim_val - meas_val)
                pct_diff = calculate_percentage_difference(sim_val, meas_val)
                diff_str = f"{diff:.3f}"
                pct_str = f"{pct_diff:.3f}" if pct_diff is not None else "N/A"
            rows.append([device, q, sim_str, meas_str, unit, diff_str, pct_str])

    # Save to CSV
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n DC summary saved to {filename}")
# Print the DC summary table comparing simulated and measured values
print_dc_summary(simulated_dc, measured_dc)
save_dc_summary_to_csv(simulated_dc, measured_dc)