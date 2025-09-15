import ltspice
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

raw_file = "Lab2-part2 b sweep.raw"
file_path = "LAB_02_DATA/NMOS_Parametric_Sweep_Lab_02_Exp_02_Excel.xlsx"

df = pd.read_excel(file_path)

Mes_I_DF = pd.read_excel(file_path, usecols=["VGS=0.00V.1"])
Mes_I_uT = array = np.array(Mes_I_DF)
Mes_I = np.delete(Mes_I_uT, 0, axis=0)
Mes_VDs_DF = pd.read_excel(file_path, usecols=["VGS=0.00V"])
Mes_VDs_uT = array = np.array(Mes_VDs_DF)
Mes_VDs = np.delete(Mes_VDs_uT, 0, axis=0)

l = ltspice.Ltspice(raw_file)
l.parse()
signals = l.variables
R1 = 1250
voltage = l.get_data('V(vo)')
voltageIN = l.get_data('V(pvdd)')

def divide_array(arr, divisor):
    if divisor == 0:
        return "Error: Division by zero is not allowed"
    return [x / divisor for x in arr]

Current = divide_array(voltage, R1)

plt.plot(voltageIN, Current, label="Simulated Current $I_D$", color="g", linestyle="-", linewidth=2)
plt.plot(Mes_VDs, Mes_I, label="Measured Current $I_D$", color="r", linestyle="-", linewidth=2)
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.title('Measured vs Simulation')
plt.legend()  # Show legend
plt.grid()
plt.show()