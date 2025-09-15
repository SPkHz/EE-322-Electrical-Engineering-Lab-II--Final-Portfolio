import matplotlib.pyplot as plt

# Data for AC Small Signal Gain RL = 10MΩ
rsig_values = ['10MΩ', '100kΩ', '10kΩ', '1kΩ']
gv_sim_rl_10M = [21.8688, 43.3046, 43.6939, 43.7333]
gv_meas_rl_10M = [7.6110, 16.2100, 20.1700, 22.3100]

# Data for AC Small Signal Gain Rsig = 1kΩ
rl_values = ['10MΩ', '100kΩ', '10kΩ', '1kΩ']
gv_sim_rsig_1k = [43.7333, 27.8597, 6.4794, 0.7839]
gv_meas_rsig_1k = [31.2000, 23.5000, 3.6500, 0.9510]

# Plotting RL = 10MΩ
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
x = range(len(rsig_values))
plt.bar(x, gv_sim_rl_10M, width=0.4, label='Simulated', align='center')
plt.bar([p + 0.4 for p in x], gv_meas_rl_10M, width=0.4, label='Measured', align='center')
plt.xticks([p + 0.2 for p in x], rsig_values)
plt.ylabel("Gain (V/V)")
plt.title("AC Gain vs Rsig (RL = 10MΩ)")
plt.legend()

# Plotting Rsig = 1kΩ
plt.subplot(1, 2, 2)
x = range(len(rl_values))
plt.bar(x, gv_sim_rsig_1k, width=0.4, label='Simulated', align='center')
plt.bar([p + 0.4 for p in x], gv_meas_rsig_1k, width=0.4, label='Measured', align='center')
plt.xticks([p + 0.2 for p in x], rl_values)
plt.ylabel("Gain (V/V)")
plt.title("AC Gain vs RL (Rsig = 1kΩ)")
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
x = range(len(rsig_values))
plt.bar(x, gv_sim_rl_10M, width=0.4, label='Simulated', align='center')
plt.bar([p + 0.4 for p in x], gv_meas_rl_10M, width=0.4, label='Measured', align='center')
plt.xticks([p + 0.2 for p in x], rsig_values)
plt.ylabel("Gain (V/V)")
plt.title("AC Gain vs Rsig (RL = 10MΩ)")
plt.legend()
plt.tight_layout()
plt.show()

# Plotting AC Gain vs RL (Rsig = 1kΩ)
plt.figure(figsize=(7, 5))
x = range(len(rl_values))
plt.bar(x, gv_sim_rsig_1k, width=0.4, label='Simulated', align='center')
plt.bar([p + 0.4 for p in x], gv_meas_rsig_1k, width=0.4, label='Measured', align='center')
plt.xticks([p + 0.2 for p in x], rl_values)
plt.ylabel("Gain (V/V)")
plt.title("AC Gain vs RL (Rsig = 1kΩ)")
plt.legend()
plt.tight_layout()
plt.show()