import numpy as np


def get_pfm_pulse(real_voltage: float) -> float:
    v_pfm = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5]  # fmt: skip
    v_real = [0.36, 0.75, 1.12, 1.48, 1.88, 2.2, 2.58, 2.92, 3.32, -0.366, -0.72, -1.1, -1.47, -1.82, -2.19, -2.56, -2.93, -3.3]  # fmt: skip
    k, b = np.polyfit(v_pfm, v_real, 1)
    return (real_voltage - b) / k


if __name__ == "__main__":
    print(get_pfm_pulse(3.5))
