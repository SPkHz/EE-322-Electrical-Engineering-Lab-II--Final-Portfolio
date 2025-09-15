# Path to the .log file
log_file_path = r'Labs/Lab-02/Lab_02_Exp_01_Sim_NMOS_SWEEP_OP.log'

# Function to extract values from the .log file
def extract_op_values_from_log(file_path):
    results = {"VGS": None, "VDS": None, "ID": None, "VOV": None, "VTH": None, "RD": None}
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                # Search for Vgs, Vds, Id, Vth, and other relevant values
                if "Vgs:" in line:
                    results["VGS"] = float(line.split()[1])  # Extract second value in the line
                elif "Vds:" in line:
                    results["VDS"] = float(line.split()[1])
                elif "Id:" in line:
                    results["ID"] = float(line.split()[1]) * 1e6  # Convert to µA
                elif "Vth:" in line:
                    results["VTH"] = float(line.split()[1])
                elif "Vdsat:" in line:
                    results["VOV"] = results["VGS"] - results["VTH"]  # Calculate VOV if VGS and VTH are available
    except Exception as e:
        print(f"Error reading the .log file: {e}")
    return results

# Call the function to extract values
op_values = extract_op_values_from_log(log_file_path)

# Display the extracted values
print("Extracted values from the .log file:")
for key, value in op_values.items():
    if value is not None:
        print(f"{key}: {value}")
    elif key == "RD":  # Ensuring RD is also printed
        print(f"{key}: {value if value != 'Not found' else 'N/A'}")

import chardet
import pandas as pd
import os

log_file_path = r'Lab_02_Exp_01_Sim_NMOS_SWEEP_OP.log'


def extract_op_values_from_log(file_path):
    results = {"VGS": None, "VDS": None, "ID": None, "VOV": None, "VTH": None}
    try:
        # Detect file encoding dynamically
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            detected_encoding = chardet.detect(raw_data)['encoding']

        # Read the file with the detected encoding
        with open(file_path, "r", encoding=detected_encoding) as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()

                # Skip lines with "0.00e+00"
                if "0.00e+00" in line:
                    continue

                if line:  # Only process non-empty lines
                    if "Vgs:" in line:
                        results["VGS"] = float(line.split(":")[1].strip())
                    elif "Vds:" in line:
                        results["VDS"] = float(line.split(":")[1].strip())
                    elif "Id:" in line and "device_current" not in line:
                        results["ID"] = float(line.split(":")[1].strip())  # Leave as is for CSV
                    elif "Vth:" in line:
                        results["VTH"] = float(line.split(":")[1].strip())
                    elif "I(Rd):" in line:
                        results["I(Rd)"] = float(line.split(":")[1].strip())

            # Calculate VOV (Overdrive Voltage)
            if results["VGS"] is not None and results["VTH"] is not None:
                results["VOV"] = results["VGS"] - results["VTH"]

    except Exception as e:
        print(f"Error reading the .log file: {e}")

    # Replace None with "Not found"
    for key in results:
        if results[key] is None:
            results[key] = "Not found"

    # Save the results to a CSV file
    try:
        save_path = os.path.dirname(file_path)
        csv_file_path = os.path.join(save_path, "Lab_02_Exp_01B_Sim_NMOS_SWEEP_OP.csv")

        results_df = pd.DataFrame([results])
        results_df.to_csv(csv_file_path, index=False)
        print(f"CSV successfully saved to {csv_file_path}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")

    return results


def format_with_si_units(value, unit=""):
    """
    Format a numerical value with SI prefixes for better readability.
    """
    if isinstance(value, str):  # Handle cases where the value is "Not found"
        return value
    if unit == "A":  # Handle current (amps) specifically
        if abs(value) >= 1:  # Value is in amperes
            return f"{value:.2f} A"
        elif abs(value) >= 1e-3:  # Value is in milliamps
            return f"{value * 1e3:.2f} mA"
        elif abs(value) >= 1e-6:  # Value is in microamps
            return f"{value * 1e6:.2f} μA"
        else:  # Value is in nanoamps
            return f"{value * 1e9:.2f} nA"
    elif unit == "V":  # Handle voltage
        if abs(value) >= 1:  # Value is in volts
            return f"{value:.2f} V"
        elif abs(value) >= 1e-3:  # Value is in millivolts
            return f"{value * 1e3:.2f} mV"
        elif abs(value) >= 1e-6:  # Value is in microvolts
            return f"{value * 1e6:.2f} μV"
    return f"{value:.2e} {unit}"  # Default scientific notation with unit


# Test the function
op_values = extract_op_values_from_log(log_file_path)

# Print the results with proper SI units
print("\nSimulated values from the LTSpice .log file:")
for key, value in op_values.items():
    if key == "ID":  # Current
        print(f"{key}: {format_with_si_units(value, 'A')}")
    else:  # Voltages (VGS, VDS, VOV, VTH)
        print(f"{key}: {format_with_si_units(value, 'V')}")