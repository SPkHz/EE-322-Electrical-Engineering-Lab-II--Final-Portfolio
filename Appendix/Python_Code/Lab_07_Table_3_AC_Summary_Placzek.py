import ltspice
import numpy as np
import math

# === Helper Functions ===
def signal_magnitude(signal):
    return (max(signal) - min(signal)) / 2

def log_dB(value):
    return 20 * math.log10(value)

# === Differential Mode ===
l_diff = ltspice.Ltspice("EE322_Lab_07_Differential_Input_Placzek.raw")
l_diff.parse()

VR_diff = l_diff.get_data('V(vr)')
Vn_diff = l_diff.get_data('V(vn)')

VRmag_diff = signal_magnitude(VR_diff)
Vnmag_diff = signal_magnitude(Vn_diff)

AD_single_R = (VRmag_diff / Vnmag_diff) / 2
AD_single_R_dB = log_dB(AD_single_R)

# === Common Mode ===
l_cm = ltspice.Ltspice("EE322_Lab_07_Common_Mode_Placzek.raw")
l_cm.parse()

VR_cm = l_cm.get_data('V(vr)')
Vn_cm = l_cm.get_data('V(vn)')
Vp_cm = l_cm.get_data('V(vp)')

VRmag_cm = signal_magnitude(VR_cm)
Vcm = (signal_magnitude(Vn_cm) + signal_magnitude(Vp_cm)) / 2

Acm_single_R = VRmag_cm / Vcm
Acm_single_R_dB = log_dB(Acm_single_R)

# === CMRR Calculation ===
CMRR = AD_single_R / Acm_single_R
CMRR_dB = log_dB(CMRR)

# === Output ===
print('------------------------------------')
print('Ad_Single (V/V):', round(AD_single_R, 5))
print('Ad_Single (dB):', round(AD_single_R_dB, 5))
print('------------------------------------')
print('Acm_Single (V/V):', round(Acm_single_R, 5))
print('Acm_Single (dB):', round(Acm_single_R_dB, 5))
print('------------------------------------')
print('CMRR (V/V):', round(CMRR, 5))
print('CMRR (dB):', round(CMRR_dB, 5))
print('------------------------------------')