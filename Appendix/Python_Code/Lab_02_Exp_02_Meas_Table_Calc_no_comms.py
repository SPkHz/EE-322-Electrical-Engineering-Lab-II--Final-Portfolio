import numpy as np
from scipy.optimize import root

# Constants
K_n1, K_n2 = 540e-6, 4 * 540e-6
K_p3, K_p4 = 176e-6, 4 * 176e-6
V_Th1, V_Th2, V_Th3, V_Th4 = 0.573, 0.573, -0.647, -0.647
V_DD = 10.0
R = 1.0361e3


def solve_vov(id_target, k):
    sol = root(lambda v_ov: (k / 2) * v_ov ** 2 - id_target, 0.1)
    return sol.x[0] if sol.success else None


def calculate_parameters(vds, id_current, r):
    id_current *= 1e-6  # µA to A
    r *= 1e3  # kΩ to Ω

    vgs1 = vds
    vov1 = solve_vov(id_current, K_n1)
    if vov1 is None: return None

    id1 = (K_n1 / 2) * vov1 ** 2
    vgs2, vds2 = vgs1, id_current * r
    vov2 = solve_vov(id_current, K_n2)
    id2 = (K_n2 / 2) * vov2 ** 2

    vsg3, vov3 = abs(V_DD - vds), solve_vov(id_current, K_p3)
    id3 = (K_p3 / 2) * vov3 ** 2
    vsg4, vov4 = vsg3, solve_vov(id_current, K_p4)
    id4 = (K_p4 / 2) * vov4 ** 2

    return {k: v * 1e6 if "ID" in k else v for k, v in {
        'VGS1': vgs1, 'VDS1': vds, 'VOV1': vov1, 'ID1': id1,
        'VGS2': vgs2, 'VDS2': vds2, 'VOV2': vov2, 'ID2': id2,
        'VSG3': vsg3, 'VOV3': vov3, 'ID3': id3,
        'VSG4': vsg4, 'VOV4': vov4, 'ID4': id4
    }.items()}


# Example
params = calculate_parameters(1.985, 506.814, 1.0361)
if params:
    print("\nCalculated Parameters:")
    for key, value in params.items():
        print(f"{key}: {value:.3f}")