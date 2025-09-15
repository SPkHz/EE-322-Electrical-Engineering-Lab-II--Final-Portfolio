import serial
import pyvisa
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Serial Configuration
serial_port = 'COM4'  # Verify the COM port
baud_rate = 9600
timeout = 1
ngu = serial.Serial(port=serial_port, baudrate=baud_rate, timeout=timeout)

# Initialize VISA Resource Manager for the DMM
rm = pyvisa.ResourceManager()

dmm = rm.open_resource('USB0::0x05E6::0x6500::04499374::INSTR') # Name of instrument
dmm.write("*RST") # Resets instrument
dmm.write('SENS:FUNC "CURR:DC"') # Set DC current measurements
dmm.write('SENS:CURR:RANG:AUTO ON') # Enables auto ranging

# Sweep Parameters
start_voltage = 0    # Starting VGS (V)
end_voltage = 4       # Ending VGS (V)
step_voltage = 0.1    # Step size (200mV)
sweep_voltages = np.arange(start_voltage, end_voltage + step_voltage, step_voltage)

# Data Storage
all_data = {}

try:
    # Reset the instrument to a safe state
    ngu.write(b'*RST\n')
    time.sleep(0.5)  # Allow some time for the reset to complete

    # Set initial voltage and current limits to safe values
    ngu.write(b'SOUR:VOLT 0\n')  # Set voltage to 0 V
    ngu.write(b'SOUR:CURR 0.02\n')  # Set current limit to 20 mA
    ngu.write(b'SENS:CURR:RANG:AUTO ON\n') # Set current range to Auto
    time.sleep(0.1)

    # Enable the output
    ngu.write(b'OUTP ON\n')
    time.sleep(0.1)

    while True:
        # Prompt for VDS
        try:
            vds = float(input("Enter VDS value (or type 'stop' to exit): "))
        except ValueError:
            print("Stopping sweeps.")
            break

        print(f"Starting sweep for VDS = {vds} V")

        # Initialize lists for this sweep
        measured_voltages = []
        currents = []
        
        # This step resets the voltage back to 0 before a new sweep begins
        ngu.write(b'SOUR:VOLT 0\n')  
        time.sleep(0.5)  # Pause to ensure voltage is set before the next sweep

        for voltage in sweep_voltages:
            # Set VGS (controlled by NGU401)
            ngu.write(f'SOUR:VOLT {voltage:.3f}\n'.encode())
            time.sleep(0.3)

            # Measure voltage
            ngu.write(b'MEAS:VOLT?\n')  # SCPI command to measure voltage
            measured_voltage = float(ngu.readline().decode().strip())
            measured_voltages.append(measured_voltage)

            # Measure current through sense ports
            try:
                dmm_current = float(dmm.query("MEAS:CURR:DC?").strip())
                currents.append(dmm_current)
            except Exception as e:
                    print(f"Error querying current from DMM: {e}")
                    currents.append(None)  # Append None for failed measurement to keep data_vds consistent
    
            time.sleep(0.2)  # Small delay for DMM to process commands

            # Print progress
            print(f"VDS: {vds:.2f} V, VGS: {measured_voltage:.3f} V, Measured Current: {dmm_current if 'dmm_current' in locals() else 'N/A'} A")


        # Save this sweep's data_vds in the dictionary
        all_data[f'VDS={vds:.2f}V'] = {'VGS': measured_voltages, 'Current': currents}

    # Disable output of NGU after all sweeps
    ngu.write(b'OUTP OFF\n')

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    ngu.close()
    dmm.close()
    
    # Plot all data_vds collected
    plt.figure()
    for vds_label, data in all_data.items():
        plt.plot(data['VGS'], data['Current']*1e3, label=f"VDS = {vds_label}") # Plot in mA

    plt.xlabel("$V_{GS}$ (V)")
    plt.ylabel("$I_{D}$ (mA)")
    plt.title("Experiment 2 - $I_{D}$ vs. $V_{GS}$")
    plt.legend()
    plt.grid(True)
    plt.show()
    plot_filename = 'NMOS_VGS_IV_Curve.png'  # You can change the file name and format as needed
    plt.savefig(plot_filename, dpi=300)

    # Prepare MultiIndex DataFrame
    multi_index_columns = []
    multi_index_data = []

    for vds_label, data in all_data.items():
        multi_index_columns.extend([(vds_label, 'VGS'), (vds_label, 'Current')])
        multi_index_data.append(data['VGS'])
        multi_index_data.append(data['Current'])

    # Convert to DataFrame
    multi_index = pd.MultiIndex.from_tuples(multi_index_columns, names=['VDS', 'Parameter'])
    formatted_data = pd.DataFrame(np.array(multi_index_data).T, columns=multi_index)

    # Save to CSV
    output_file = 'VGS_NMOS_Parametric_Sweep.csv'
    formatted_data.to_csv(output_file, index=False)
    print(f"All sweeps completed. Data saved to '{output_file}'.")