import serial
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Serial Configuration
serial_port = 'COM4'  # Verify the COM port
baud_rate = 9600
timeout = 1
ngu = serial.Serial(port=serial_port, baudrate=baud_rate, timeout=timeout)

# Sweep Parameters
start_voltage = 0    # Starting VDS (V)
end_voltage = 8       # Ending VDS (V)
step_voltage = 0.1    # Step size (100mV)
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
        # Prompt for VGS from the user in the Python Console
        try:
            vgs = float(input("Enter VGS value (or type 'stop' to exit): "))
        except ValueError:
            print("Stopping sweeps.")
            break

        print(f"Starting sweep for VGS = {vgs} V")

        # Initialize lists for this sweep
        measured_voltages = []
        currents = []
        
        # This step resets the voltage back to 0 before a new sweep begins
        ngu.write(b'SOUR:VOLT 0\n')  
        time.sleep(0.5)  # Pause to ensure voltage is set before the next sweep

        for voltage in sweep_voltages:
            # Set VDS (controlled by NGU401)
            ngu.write(f'SOUR:VOLT {voltage:.3f}\n'.encode())
            time.sleep(0.3) # The time inbetween sweep points

            # Measure current
            ngu.write(b'MEAS:CURR?\n')
            measured_current = float(ngu.readline().decode().strip())
            currents.append(measured_current)
            
            # Measure voltage
            ngu.write(b'MEAS:VOLT?\n')  # SCPI command to measure voltage
            measured_voltage = float(ngu.readline().decode().strip())
            measured_voltages.append(measured_voltage)
            
            # Print progress
            print(f"VGS: {vgs:.2f} V, VDS: {measured_voltage:.3f} V, Measured Current: {measured_current:.6f} A")

        # Save this sweep's data_vds in the dictionary
        all_data[f'VGS={vgs:.2f}V'] = {'VDS': measured_voltages, 'Current': currents}

    # Disable output after all sweeps
    ngu.write(b'OUTP OFF\n')

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    ngu.close() # This closes communication with the instrument
    
    # Plot all data_vds collected
    plt.figure()
    for vgs_label, data in all_data.items():
        plt.plot(data['VDS'], data['Current']*1e3, label=f"VGS = {vgs_label}") # Scaling current to mA

    plt.xlabel("$V_{DS}$ (V)")
    plt.ylabel("$I_{D}$ (mA)")
    plt.title("Experiment 1 - $I_{D}$ vs. $V_{DS}$")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Prepare MultiIndex DataFrame
    multi_index_columns = []
    multi_index_data = []

    for vgs_label, data in all_data.items():
        multi_index_columns.extend([(vgs_label, 'VDS'), (vgs_label, 'Current')])
        multi_index_data.append(data['VDS'])
        multi_index_data.append(data['Current'])

    # Convert to DataFrame
    multi_index = pd.MultiIndex.from_tuples(multi_index_columns, names=['VGS', 'Parameter'])
    formatted_data = pd.DataFrame(np.array(multi_index_data).T, columns=multi_index)

    # Save to CSV
    output_file = 'NMOS_Parametric_Sweep.csv'
    formatted_data.to_csv(output_file, index=False)
    print(f"All sweeps completed. Data saved to '{output_file}'.")