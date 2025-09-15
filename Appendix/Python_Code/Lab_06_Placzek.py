import ltspice
import numpy as np
import math
import re
import chardet

########################## Helpers ########################
def percent_diff(sim, meas):
    try:
        return abs((sim - meas) / meas) * 100 if meas != 0 else None
    except:
        return None

def format_value(val):
    return f"{val:.4f}" if isinstance(val, float) else "N/A"

########################## AC Simulation - Single Ended ########################
def process_single_ended(filename):
    l = ltspice.Ltspice(filename)
    l.parse()

    VR = l.get_data('V(vr)')
    VL = l.get_data('V(vl)')
    Vn = l.get_data('V(vn)')
    Vp = l.get_data('V(vp)')

    VRmag = (max(VR) - min(VR)) / 2
    Vnmag = (max(Vn) - min(Vn)) / 2
    VLmag = (max(VL) - min(VL)) / 2
    Vpmag = (max(Vp) - min(Vp)) / 2

    AD_Single_R = (VRmag / Vnmag) / 2
    AD_Single_R_dB = 20 * math.log10(AD_Single_R)

    return {
        "Ad": AD_Single_R,
        "Ad_dB": AD_Single_R_dB
    }

########################## AC Simulation - Common Mode ########################
def process_common_mode(filename):
    l = ltspice.Ltspice(filename)
    l.parse()

    VR = l.get_data('V(vr)')
    VL = l.get_data('V(vl)')
    Vn = l.get_data('V(vn)')
    Vp = l.get_data('V(vp)')

    VRmag = (max(VR) - min(VR)) / 2
    Vnmag = (max(Vn) - min(Vn)) / 2
    VLmag = (max(VL) - min(VL)) / 2
    Vpmag = (max(Vp) - min(Vp)) / 2
    Vcm = (Vnmag + Vpmag) / 2
    Vout = (VRmag - VLmag) / 2

    Acm_Single_R = VRmag / Vcm
    Acm_Single_R_dB = 20 * math.log10(Acm_Single_R)

    AD_Double = (VRmag + VLmag) / (2 * Vnmag)
    AD_Double_dB = 20 * math.log10(AD_Double)

    Acm_Double = abs(Vout / Vcm)
    Acm_Double_dB = 20 * math.log10(Acm_Double)

    CMRR_Single_dB = 20 * math.log10((VRmag / Vnmag) / 2) - Acm_Single_R_dB
    CMRR_Double_dB = AD_Double_dB - Acm_Double_dB

    return {
        "Acm": Acm_Single_R,
        "Acm_dB": Acm_Single_R_dB,
        "CMRR_dB": CMRR_Single_dB,
        "Ad_Double": AD_Double,
        "Ad_Double_dB": AD_Double_dB,
        "Acm_Double": Acm_Double,
        "Acm_Double_dB": Acm_Double_dB,
        "CMRR_Double_dB": CMRR_Double_dB
    }

########################## DC Operating Point Parsing ########################
def extract_simulated_dc_points(file_path):
    simulated = {"Q1": {}, "Q2": {}}
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            encoding = chardet.detect(raw_data)['encoding']

        with open(file_path, 'r', encoding=encoding) as file:
            content = file.read()

        id_match = re.search(r'Id:\s*([-\d.e+]+)\s+([-\d.e+]+)', content)
        vgs_match = re.search(r'Vgs:\s*([-\d.e+]+)\s+([-\d.e+]+)', content)
        vds_match = re.search(r'Vds:\s*([-\d.e+]+)\s+([-\d.e+]+)', content)
        vth_match = re.search(r'Vth:\s*([-\d.e+]+)\s+([-\d.e+]+)', content)

        if id_match and vgs_match and vds_match and vth_match:
            # Q1
            simulated["Q1"]["ID"] = float(id_match.group(1)) * 1e6
            simulated["Q1"]["VGS"] = float(vgs_match.group(1))
            simulated["Q1"]["VDS"] = float(vds_match.group(1))
            simulated["Q1"]["VTH"] = float(vth_match.group(1))
            simulated["Q1"]["VOV"] = abs(simulated["Q1"]["VGS"] - simulated["Q1"]["VTH"])
            simulated["Q1"]["VG"] = simulated["Q1"]["VGS"]
            simulated["Q1"]["VD"] = simulated["Q1"]["VDS"]
            simulated["Q1"]["VS"] = 0

            # Q2
            simulated["Q2"]["ID"] = float(id_match.group(2)) * 1e6
            simulated["Q2"]["VGS"] = float(vgs_match.group(2))
            simulated["Q2"]["VDS"] = float(vds_match.group(2))
            simulated["Q2"]["VTH"] = float(vth_match.group(2))
            simulated["Q2"]["VOV"] = abs(simulated["Q2"]["VGS"] - simulated["Q2"]["VTH"])
            simulated["Q2"]["VG"] = simulated["Q2"]["VGS"]
            simulated["Q2"]["VD"] = simulated["Q2"]["VDS"]
            simulated["Q2"]["VS"] = 0

    except Exception as e:
        print(f"Error parsing log file: {e}")

    return simulated

########################## Table Printers ########################
def print_dc_summary(simulated, measured):
    print("\n========== DC Summary Table ==========")
    headers = f"{'Device':<6} | {'Quantity':<6} | {'Simulated':>10} | {'Measured':>10} | Units"
    print(headers)
    print("-" * len(headers))

    for device in ["Q1", "Q2"]:
        quantities = [
            ("ID", "μA"),
            ("VOV", "V"),
            ("VG", "V"),
            ("VD", "V"),
            ("VS", "V")
        ]
        for q, unit in quantities:
            sim_val = simulated[device].get(q, "N/A")
            meas_val = measured[device].get(q, "N/A")
            sim_str = f"{sim_val:.3f}" if isinstance(sim_val, float) else sim_val
            meas_str = f"{meas_val:.3f}" if isinstance(meas_val, float) else meas_val
            print(f"{device:<6} | {q:<6} | {sim_str:>10} | {meas_str:>10} | {unit}")
    print("=" * len(headers))


def print_ac_summary_table(title, data):
    print(f"\n========== {title} ==========")
    headers = f"{'Quantity':<8} | {'Simulated':>10} | {'Measured':>10} | {'Diff (%)':>10} | Units"
    print(headers)
    print("-" * len(headers))

    for row in data:
        quantity, sim_val, meas_val, units = row
        diff = percent_diff(sim_val, meas_val) if meas_val is not None else None
        sim_str = f"{sim_val:.4f}" if isinstance(sim_val, float) else "N/A"
        meas_str = f"{meas_val:.4f}" if isinstance(meas_val, float) else "N/A"
        diff_str = f"{diff:.2f}%" if diff is not None else "N/A"
        print(f"{quantity:<8} | {sim_str:>10} | {meas_str:>10} | {diff_str:>10} | {units}")
    print("=" * len(headers))

########################## MAIN EXECUTION ########################
# File paths
single_filename = "Lab_6_180_Deg_Quadrature_Placzek.raw"
common_filename = "Lab_6_Common_Mode_Placzek.raw"
log_file_path = '/opt/miniconda3/envs/pycharm-env/EE322_ee_Lab_II/Labs/Lab-06/EE322_Lab_6_Measured_Values_Placzek.log'

# AC analysis
single_results = process_single_ended(single_filename)
common_results = process_common_mode(common_filename)

# Measured AC values
measured_ac = {
    "Ad": 6.205,
    "Ad_dB": 15.801,
    "Acm": 0.499,
    "Acm_dB": -5.924,
    "CMRR_dB": 21.724,
    "Ad_Double": 6.544,
    "Ad_Double_dB": 16.319,
    "Acm_Double": 0.342,
    "Acm_Double_dB": -9.312,
    "CMRR_Double_dB": 25.631
}

# DC analysis
simulated_dc = extract_simulated_dc_points(log_file_path)
measured_dc = {
    "Q1": {"ID": 483.110, "VOV": 1.141, "VG": 0.000, "VD": 2.896, "VS": -1.820},
    "Q2": {"ID": 524.043, "VOV": 1.401, "VG": 0.000, "VD": 2.011, "VS": -1.819}
}

# AC Summary Tables
single_ended_data = [
    ("Ad", single_results["Ad"], measured_ac["Ad"], "V/V"),
    ("Ad", single_results["Ad_dB"], measured_ac["Ad_dB"], "dB"),
    ("Acm", common_results["Acm"], measured_ac["Acm"], "V/V"),
    ("Acm", common_results["Acm_dB"], measured_ac["Acm_dB"], "dB"),
    ("CMRR", common_results["CMRR_dB"], measured_ac["CMRR_dB"], "dB"),
]

differential_data = [
    ("Ad", common_results["Ad_Double"], measured_ac["Ad_Double"], "V/V"),
    ("Ad", common_results["Ad_Double_dB"], measured_ac["Ad_Double_dB"], "dB"),
    ("Acm", common_results["Acm_Double"], measured_ac["Acm_Double"], "V/V"),
    ("Acm", common_results["Acm_Double_dB"], measured_ac["Acm_Double_dB"], "dB"),
    ("CMRR", common_results["CMRR_Double_dB"], measured_ac["CMRR_Double_dB"], "dB"),
]

# Print everything
print_dc_summary(simulated_dc, measured_dc)
print_ac_summary_table("AC Summary - Single Ended", single_ended_data)
print_ac_summary_table("AC Summary - Differential", differential_data)