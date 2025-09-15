import skrf as rf
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

open_standard = rf.Network('Open.s2p')
short_standard = rf.Network('Short.s2p')
thru_standard = rf.Network('Thru.s2p')
dut = rf.Network('hpf.s2p')

def deembed_osp(dut, open_std, short_std, thru_std):
    thru_half = thru_std.copy()
    thru_half.s = np.zeros_like(thru_std.s)
    for i in range(len(thru_std.f)):
        s = thru_std.s[i]
        s_3d = s.reshape(1, *s.shape)
        t = rf.s2t(s_3d)
        t_half = scipy.linalg.sqrtm(t[0])
        t_half_3d = t_half.reshape(1, *t_half.shape)
        s_half = rf.t2s(t_half_3d)[0]
        thru_half.s[i] = s_half

    y_open = open_std.y
    z_short = short_std.z
    y_parasitic = y_open
    y_short = short_std.y
    y_short_corrected = y_short - y_parasitic
    z_parasitic = rf.network.y2z(y_short_corrected)

    dut_y = dut.y
    dut_y_corrected = dut_y - y_parasitic
    dut_z = rf.network.y2z(dut_y_corrected)
    dut_z_corrected = dut_z - z_parasitic

    dut_corrected = rf.Network(s=rf.z2s(dut_z_corrected),
                               frequency=dut.frequency)
    return dut_corrected

hpf_deembedded = deembed_osp(dut, open_standard, short_standard, thru_standard)
hpf_deembedded.write_touchstone('hpf_deembedded.s2p')

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.title('HPF - OST - S21 - Transmission')
dut.plot_s_db(m=0, n=1, label='Original')
hpf_deembedded.plot_s_db(m=0, n=1, label='OST De-embedded')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.title('HPF - OST De-Embed - S11 - Reflection')
dut.plot_s_db(m=0, n=0, label='Original')
hpf_deembedded.plot_s_db(m=0, n=0, label='OST De-embedded')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('hpf_deembedded_comparison.png')
plt.show()