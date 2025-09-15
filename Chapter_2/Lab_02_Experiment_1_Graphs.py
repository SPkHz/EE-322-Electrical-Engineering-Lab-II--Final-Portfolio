import ltspice
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#-------------------SHIPING AND RECIVING-------------------------------------#
raw_file = "Lab_02_Exp_01A_Sim_NMOS_SWEEP_MEAS.raw"
file_path = "Experiment_1_Part_A_NMOS_Sweep_data.csv"

df = pd.read_excel(file_path)  # Reads the first sheet by default

 #Display the first few rows
print(df.head())


# ---------------------Excle data procsesing
Mes_I_DF = pd.read_excel(file_path, usecols=["VGS=0.00V.1"])  # Replace with actual column names
Mes_I_uT = array = np.array(Mes_I_DF)
Mes_I = np.delete(Mes_I_uT, 0, axis=0)
Mes_VDs_DF = pd.read_excel(file_path, usecols=["VGS=0.00V"])  # Replace with actual column names
Mes_VDs_uT = array = np.array(Mes_VDs_DF)
Mes_VDs = np.delete(Mes_VDs_uT, 0, axis=0)

#--------------Ltspice data procseing
l = ltspice.Ltspice(raw_file)
l.parse()  # Parse the file

signals = l.variables
print("Available signals:", signals)
Ird = l.get_data('I(Rd)')  # Use an available variable name
voltageIN = l.get_data('V(vdd)')  # Use an available variable name


# Plot the signal


plt.plot(voltageIN, Ird, label="Simulated Current", color="g", linestyle="-", linewidth=2)

plt.plot(Mes_VDs, Mes_I, label="Messured Current", color="r", linestyle="-", linewidth=2)
plt.xlabel('Voltage (V)')
plt.ylabel('Currnet (A)')
plt.title('Measured vs Simulation')
plt.legend()  # Show legend

plt.grid()
plt.show()