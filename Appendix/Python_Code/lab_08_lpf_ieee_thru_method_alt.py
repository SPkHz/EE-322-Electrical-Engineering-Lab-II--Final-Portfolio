import skrf as rf
import ieee_p370_2x_thru_alt
# from ieee_p370_2x_thru_alt import deembed_s_params
import numpy as np

def deembed_s_params(s_fixture_dut_fixture, s_side1, s_side2):
    """
    De-embeds the S-parameters of a DUT from measurements that include test fixtures.

    Parameters
    ----------
    s_fixture_dut_fixture : rf.Network
        S-parameter network object of the DUT with fixtures
    s_side1 : rf.Network
        S-parameter network object of the error box for port 1
    s_side2 : rf.Network
        S-parameter network object of the error box for port 2

    Returns
    -------
    s_deembedded_dut : rf.Network
        De-embedded S-parameter network object of the DUT
    """
    # Create T-parameters for the error boxes
    t_side1 = rf.s2t(s_side1.s)
    t_side2 = rf.s2t(s_side2.s)

    # Create T-parameters for the fixture+DUT+fixture
    t_fixture_dut_fixture = rf.s2t(s_fixture_dut_fixture.s)

    # De-embed the DUT
    t_dut = np.zeros(t_fixture_dut_fixture.shape, dtype=complex)

    # T_DUT = inv(T_side1) * T_fixture_dut_fixture * inv(T_side2)
    for i in range(len(t_fixture_dut_fixture)):
        t_dut[i] = (
            np.linalg.inv(t_side1[i])
            @ t_fixture_dut_fixture[i]
            @ np.linalg.inv(t_side2[i])
        )

    # Convert back to S-parameters
    s_deembedded_dut = s_fixture_dut_fixture.copy()
    s_deembedded_dut.s = rf.t2s(t_dut)

    return s_deembedded_dut

# 1. Load your 2x-thru calibration measurement
s_2xthru = rf.Network('Thru.s2p')

# 2. Generate the error boxes (fixture models) from the 2x-thru
# Modified to use use_ifft=False to avoid the problematic IFFT calculation
s_side1, s_side2 = ieee_p370_2x_thru_alt.ieee_p370_2x_thru_alt(s_2xthru, use_ifft=False)

# 3. Load your DUT measurement (with fixtures)
s_fixture_dut_fixture = rf.Network('lpf.s2p')

# 4. De-embed the fixture effects to get the actual DUT response
s_deembedded_dut = deembed_s_params(s_fixture_dut_fixture, s_side1, s_side2)

# 5. Save the de-embedded DUT S-parameters to a file
s_deembedded_dut.write_touchstone('lpf_ieee_deembedded_dut.s2p')

# Optional: Plot the results to verify
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.title('LPF S-parameters Before and After De-embedding')

# Plot S11 before and after de-embedding
plt.plot(s_fixture_dut_fixture.f/1e9, 20*np.log10(np.abs(s_fixture_dut_fixture.s[:,0,0])),
         label='S11 with fixtures')
plt.plot(s_deembedded_dut.f/1e9, 20*np.log10(np.abs(s_deembedded_dut.s[:,0,0])),
         label='S11 de-embedded')

# Plot S21 before and after de-embedding
plt.plot(s_fixture_dut_fixture.f/1e9, 20*np.log10(np.abs(s_fixture_dut_fixture.s[:,1,0])),
         label='S21 with fixtures')
plt.plot(s_deembedded_dut.f/1e9, 20*np.log10(np.abs(s_deembedded_dut.s[:,1,0])),
         label='S21 de-embedded')

plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid(True)
plt.show()