import numpy as np
import skrf as rf
import scipy.signal as signal
import scipy.interpolate as interp
import warnings
from typing import Tuple, Optional, List, Union


def ieee_p370_2x_thru_alt(
    s_2xthru: rf.Network, use_ifft: bool = True
) -> Tuple[rf.Network, rf.Network]:
    """
    Creates error boxes from a test fixture 2x thru.

    Parameters
    ----------
    s_2xthru : skrf.Network
        An s parameter network object of the 2x thru
    use_ifft : bool, optional
        Whether to use the time domain method (True) or impedance method (False), by default True

    Returns
    -------
    s_side1 : skrf.Network
        An s parameter network object of the error box representing the half of the 2x thru connected to port 1
    s_side2 : skrf.Network
        An s parameter network object of the error box representing the half of the 2x thru connected to port 2
    """
    f = s_2xthru.f  # Frequency array
    s = s_2xthru.s  # S-parameters (shape: freq_points x 2 x 2)

    # ----------- main --------------------
    if use_ifft:
        # strip DC point if one exists
        if f[0] == 0:
            warnings.warn(
                "DC point detected. An interpolated DC point will be included in the errorboxes."
            )
            flag_DC = 1
            fold = f.copy()
            f = f[1:]
            s = s[1:, :, :]
        else:
            flag_DC = 0

        # interpolate S-parameters if the frequency vector is not acceptable
        if f[1] - f[0] != f[0]:
            # set the flag
            flag_df = 1
            warnings.warn(
                "Non-uniform frequency vector detected. A spline interpolated S-parameter matrix will be created for this calculation. The output results will be re-interpolated to the original vector."
            )
            fold = f.copy()

            df = f[1] - f[0]
            projected_n = round(f[-1] / f[0])

            if projected_n <= 10000:
                if f[-1] % f[0] == 0:
                    fnew = np.arange(f[0], f[-1] + f[0], f[0])
                else:
                    fnew = np.arange(f[0], f[-1] - (f[-1] % f[0]) + f[0], f[0])
            else:
                new_df = f[-1] / 10000
                fnew = np.arange(new_df, f[-1] + new_df, new_df)
                print(f"interpolating from {new_df}Hz to {f[-1]}Hz with 10000 points.")

            snew = np.zeros((len(fnew), 2, 2), dtype=complex)
            for i in range(2):
                for j in range(2):
                    snew[:, i, j] = interp.interp1d(f, s[:, i, j], kind="cubic")(fnew)

            s = snew
            f = fnew
        else:
            flag_df = 0

        n = len(f)
        s11 = s[:, 0, 0]

        # get e001 and e002
        # e001
        s21 = s[:, 1, 0]
        dcs21 = dc_interp(s21, f)
        t21 = np.fft.fftshift(
            np.fft.ifft(make_symmetric(np.vstack([dcs21, s21])), axis=0)
        )
        x = np.argmax(np.abs(t21))

        dcs11 = dc(s11, f)
        t11 = np.fft.fftshift(
            np.fft.ifft(make_symmetric(np.vstack([dcs11, s11])), axis=0)
        )
        step11 = make_step(t11)
        z11 = -50 * (step11 + 1) / (step11 - 1)
        z11x = z11[x]

        # Convert to new reference impedance
        temp = rf.Network(f=f, s=s, z0=50)
        temp.renormalize(z11x)
        sr = temp.s

        s11r = sr[:, 0, 0]
        s21r = sr[:, 1, 0]
        s12r = sr[:, 0, 1]
        s22r = sr[:, 1, 1]

        dcs11r = dc(s11r, f)
        t11r = np.fft.fftshift(
            np.fft.ifft(make_symmetric(np.vstack([dcs11r, s11r])), axis=0)
        )
        t11r[x:] = 0
        e001 = np.fft.fft(np.fft.ifftshift(t11r))
        e001 = e001[1 : n + 1]

        dcs22r = dc(s22r, f)
        t22r = np.fft.fftshift(
            np.fft.ifft(make_symmetric(np.vstack([dcs22r, s22r])), axis=0)
        )
        t22r[x:] = 0
        e002 = np.fft.fft(np.fft.ifftshift(t22r))
        e002 = e002[1 : n + 1]

        # calc e111 and e112
        e111 = (s22r - e002) / s12r
        e112 = (s11r - e001) / s21r

        # calc e01
        k = 1
        test = k * np.sqrt(s21r * (1 - e111 * e112))
        e01 = np.zeros(n, dtype=complex)
        for i in range(n):
            if i > 0:
                if np.angle(test[i]) - np.angle(test[i - 1]) > 0:
                    k = -1 * k
            e01[i] = k * np.sqrt(s21r[i] * (1 - e111[i] * e112[i]))

        # calc e10
        k = 1
        test = k * np.sqrt(s12r * (1 - e111 * e112))
        e10 = np.zeros(n, dtype=complex)
        for i in range(n):
            if i > 0:
                if np.angle(test[i]) - np.angle(test[i - 1]) > 0:
                    k = -1 * k
            e10[i] = k * np.sqrt(s12r[i] * (1 - e111[i] * e112[i]))

        # S-parameters are setup correctly
        if flag_DC == 0 and flag_df == 0:
            fixture_model_1r = np.zeros((n, 2, 2), dtype=complex)
            fixture_model_1r[:, 0, 0] = e001
            fixture_model_1r[:, 1, 0] = e01
            fixture_model_1r[:, 0, 1] = e01
            fixture_model_1r[:, 1, 1] = e111

            fixture_model_2r = np.zeros((n, 2, 2), dtype=complex)
            fixture_model_2r[:, 1, 1] = e002
            fixture_model_2r[:, 0, 1] = e10
            fixture_model_2r[:, 1, 0] = e10
            fixture_model_2r[:, 0, 0] = e112

        else:  # S-parameters are not setup correctly
            if flag_DC == 1:  # DC Point was included in the original file
                fixture_model_1r = np.zeros((n + 1, 2, 2), dtype=complex)
                fixture_model_1r[1:, 0, 0] = e001
                fixture_model_1r[0, 0, 0] = dc_interp(fixture_model_1r[1:, 0, 0], f)
                fixture_model_1r[1:, 1, 0] = e01
                fixture_model_1r[0, 1, 0] = dc_interp(fixture_model_1r[1:, 1, 0], f)
                fixture_model_1r[1:, 0, 1] = e01
                fixture_model_1r[0, 0, 1] = dc_interp(fixture_model_1r[1:, 0, 1], f)
                fixture_model_1r[1:, 1, 1] = e111
                fixture_model_1r[0, 1, 1] = dc_interp(fixture_model_1r[1:, 1, 1], f)

                fixture_model_2r = np.zeros((n + 1, 2, 2), dtype=complex)
                fixture_model_2r[1:, 1, 1] = e002
                fixture_model_2r[0, 0, 0] = dc_interp(fixture_model_2r[1:, 0, 0], f)
                fixture_model_2r[1:, 0, 1] = e10
                fixture_model_2r[0, 1, 0] = dc_interp(fixture_model_2r[1:, 1, 0], f)
                fixture_model_2r[1:, 1, 0] = e10
                fixture_model_2r[0, 0, 1] = dc_interp(fixture_model_2r[1:, 0, 1], f)
                fixture_model_2r[1:, 0, 0] = e112
                fixture_model_2r[0, 1, 1] = dc_interp(fixture_model_2r[1:, 1, 1], f)
                f = np.concatenate(([0], f))
            else:  # DC Point wasn't included in the original file, but the DF was not the same as f[0]
                fixture_model_1r = np.zeros((n, 2, 2), dtype=complex)
                fixture_model_1r[:, 0, 0] = e001
                fixture_model_1r[:, 1, 0] = e01
                fixture_model_1r[:, 0, 1] = e01
                fixture_model_1r[:, 1, 1] = e111

                fixture_model_2r = np.zeros((n, 2, 2), dtype=complex)
                fixture_model_2r[:, 1, 1] = e002
                fixture_model_2r[:, 0, 1] = e10
                fixture_model_2r[:, 1, 0] = e10
                fixture_model_2r[:, 0, 0] = e112

            if flag_df == 1:  # if df was different from f[0]
                # save the current error boxes
                fixture_model_1r_temp = fixture_model_1r.copy()
                fixture_model_2r_temp = fixture_model_2r.copy()
                # initialize the new errorboxes
                fixture_model_1r = np.zeros((len(fold), 2, 2), dtype=complex)
                fixture_model_2r = np.zeros((len(fold), 2, 2), dtype=complex)
                # interpolate the errorboxes to the original frequency vector
                for i in range(2):
                    for j in range(2):
                        fixture_model_1r[:, i, j] = interp.interp1d(
                            f, fixture_model_1r_temp[:, i, j], kind="cubic"
                        )(fold)
                        fixture_model_2r[:, i, j] = interp.interp1d(
                            f, fixture_model_2r_temp[:, i, j], kind="cubic"
                        )(fold)

            # replace the vector used for the calculation with the original vector.
            f = fold

        # create the S-parameter objects for the errorboxes
        s_fixture_model_r1 = rf.Network(f=f, s=fixture_model_1r, z0=z11x)
        s_fixture_model_r2 = rf.Network(f=f, s=fixture_model_2r, z0=z11x)

        # renormalize the S-parameter errorboxes to the original reference impedance (assumed to be 50)
        s_side1 = s_fixture_model_r1.copy()
        s_side1.renormalize(50)
        s_side2 = s_fixture_model_r2.copy()
        s_side2.renormalize(50)
    else:
        # Use impedance method instead of time domain method
        # Convert S-parameters to Z-parameters
        z = rf.s2z(s_2xthru.s, s_2xthru.z0)
        ZL = np.zeros(z.shape, dtype=complex)
        ZR = np.zeros(z.shape, dtype=complex)

        for i in range(len(z)):
            ZL[i, 0, 0] = z[i, 0, 0] + z[i, 1, 0]
            ZL[i, 0, 1] = 2 * z[i, 1, 0]
            ZL[i, 1, 0] = 2 * z[i, 1, 0]
            ZL[i, 1, 1] = 2 * z[i, 1, 0]

            ZR[i, 0, 0] = 2 * z[i, 0, 1]
            ZR[i, 0, 1] = 2 * z[i, 0, 1]
            ZR[i, 1, 0] = 2 * z[i, 0, 1]
            ZR[i, 1, 1] = z[i, 0, 1] + z[i, 1, 1]

        # Convert Z-parameters back to S-parameters
        SL = rf.z2s(ZL, s_2xthru.z0)
        SR = rf.z2s(ZR, s_2xthru.z0)

        s_side1 = rf.Network(f=f, s=SL, z0=50)
        s_side2 = rf.Network(f=f, s=SR, z0=50)

    return s_side1, s_side2


# ------------- supporting functions ------------------


def make_symmetric(nonsymmetric):
    """
    Takes the nonsymmetric frequency domain input and makes it symmetric.
    The function assumes the DC point is in the nonsymmetric data.

    Parameters
    ----------
    nonsymmetric : ndarray
        Nonsymmetric frequency domain data

    Returns
    -------
    symmetric : ndarray
        Symmetric frequency domain data
    """
    symmetric_abs = np.vstack(
        [np.abs(nonsymmetric), np.flip(np.abs(nonsymmetric[1:]), axis=0)]
    )
    symmetric_ang = np.vstack(
        [np.angle(nonsymmetric), -np.flip(np.angle(nonsymmetric[1:]), axis=0)]
    )
    symmetric = symmetric_abs * np.exp(1j * symmetric_ang)
    return symmetric


def make_step(impulse):
    """
    Creates a step response from an impulse response.

    Parameters
    ----------
    impulse : ndarray
        Impulse response

    Returns
    -------
    step : ndarray
        Step response
    """
    ustep = np.ones(len(impulse))
    step = np.convolve(ustep, impulse)
    step = step[: len(impulse)]
    return step


def dc(s, f):
    """
    Calculates the DC point for S-parameters.

    Parameters
    ----------
    s : ndarray
        S-parameter data
    f : ndarray
        Frequency data

    Returns
    -------
    DCpoint : float
        DC point value
    """
    DCpoint = 0.002  # seed for the algorithm
    err = 1  # error seed
    allowedError = 1e-12  # allowable error
    cnt = 0
    df = f[1] - f[0]
    n = len(f)
    t = np.linspace(-1 / df, 1 / df, n * 2 + 1)
    ts = np.argmin(np.abs(t - (-3e-9)))
    Hr = com_receiver_noise_filter(f, f[-1] / 2)

    while err > allowedError:
        h1 = make_step(
            np.fft.fftshift(np.fft.ifft(make_symmetric(np.vstack([DCpoint, s * Hr]))))
        )
        h2 = make_step(
            np.fft.fftshift(
                np.fft.ifft(make_symmetric(np.vstack([DCpoint + 0.001, s * Hr])))
            )
        )
        m = (h2[ts] - h1[ts]) / 0.001
        b = h1[ts] - m * DCpoint
        DCpoint = (0 - b) / m
        err = np.abs(h1[ts] - 0)
        cnt += 1

    return DCpoint


def com_receiver_noise_filter(f, fr):
    """
    Receiver filter in COM defined by eq 93A-20.

    Parameters
    ----------
    f : ndarray
        Frequency data
    fr : float
        Reference frequency

    Returns
    -------
    Hr : ndarray
        Filter response
    """
    fdfr = f / fr
    Hr = 1.0 / (1 - 3.414214 * (fdfr) ** 2 + fdfr**4 + 1j * 2.613126 * (fdfr - fdfr**3))
    return Hr


def dc_interp(sin, f):
    """
    Enforces symmetry upon the first 10 points and interpolates the DC point.

    Parameters
    ----------
    sin : ndarray
        Input data
    f : ndarray
        Frequency data

    Returns
    -------
    dc : float
        DC point value
    """
    sp = sin[:10]
    fp = f[:10]

    snp = np.concatenate([np.conj(np.flip(sp)), sp])
    fnp = np.concatenate([-np.flip(fp), fp])
    fnew = np.concatenate([-np.flip(fp), [0], fp])
    snew = interp.interp1d(fnp, snp, kind="cubic")(fnew)
    dc = np.real(snew[len(sp)])

    return dc


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
    t_side1 = s_side1.s2t
    t_side2 = s_side2.s2t

    # Create T-parameters for the fixture+DUT+fixture
    t_fixture_dut_fixture = s_fixture_dut_fixture.s2t

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