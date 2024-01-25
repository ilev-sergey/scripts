import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt, sosfiltfilt


def vfit(fs, data, doPlot=False):  # -> fit params # TODO: add logging
    def iter(pguess, s, data):
        # print(f"begin: {pguess=}")

        pguess = (
            [pguess]
            if type(pguess) == np.complex128 or type(pguess) == np.float64
            else pguess
        )

        lhs = np.zeros((len(fs), 4 * len(pguess) + 2), dtype=complex)

        # print(f"{fs.shape=},{len(pguess)=}")
        # print(f"{lhs[:, index].shape=}, {data.shape=}")
        # print(f"{s.shape=}, {pguess=}, {data.shape=}")

        for i in range(len(pguess)):
            index = 4 * i
            real = 1.0 / (s - pguess[i]) + 1.0 / (s - np.conj(pguess[i]))
            imag = 1j / (s - pguess[i]) - 1j / (s - np.conj(pguess[i]))

            lhs[:, 4 * i] = real
            lhs[:, 4 * i + 1] = imag
            lhs[:, 4 * i + 2] = -data * real
            lhs[:, 4 * i + 3] = -data * imag

        # print(f"lhs after cycle {np.real(lhs[0,0])=}")
        lhs[:, -2] = 1
        lhs[:, -1] = s

        rhs = data

        lhs = np.vstack((np.real(lhs), np.imag(lhs)))
        rhs = np.concatenate((np.real(rhs), np.imag(rhs)))

        # print(f"{lhs.shape=}, {rhs.shape=}")
        res = np.linalg.pinv(lhs) @ rhs

        # print(f" {res.shape=}, {res=}")
        arr1 = np.array(
            [[np.real(pguess), np.imag(pguess)], [-np.imag(pguess), np.real(pguess)]]
        ).reshape(2, 2)
        arr2 = np.array([[2, 0]]).reshape(2, 1)
        arr3 = np.array([[res[2], res[3]]])
        # print(f"{arr1.shape=}, {arr2.shape=}, {arr3.shape=}")
        rm = arr1 - arr2 @ arr3

        poles = np.linalg.eigvals(rm)

        cs = res[0] + 1j * res[1]

        d = res[-2]
        h = res[-1]
        # print(f"end: {type(poles), poles.shape}")

        return poles, cs, d, h

    # Ensure fs and data are one-dimensional arrays
    # fs = np.asarrayfs)
    # print(f"{fs.shape=}, {data.shape=}")  # s0 1, s(1,), dfilt(1, 4096)
    s = 1j * fs
    fc = np.mean(fs)

    pguess = [-fc / 100 + 1j * fc]

    s0 = pguess

    niters = 30

    # sos = butter(8, 0.1, output="sos")
    # dfilt = sosfiltfilt(sos, data)
    # print(f"{data.shape=}")
    fb, fa = butter(8, 0.1)  # checked
    dfilt = filtfilt(fb, fa, data, padlen=3 * (max(len(fb), len(fa)) - 1))  # checked

    # print(f"{dfilt[0]=:.8E},\n {data[0]=:.8}")

    # print(f"s0{s0.__len__()}, s{s.shape}, dfilt{dfilt.shape}")

    for i in range(niters):
        # print(f"cycle: {s0=}")
        # print(f"i={i+1}: {s0=}\n \n")
        result = iter(pguess=s0, s=s, data=dfilt)
        # print(f"{result[0]=}")
        s0, c, D, h = result
        # print(f"{s0=}")
        s0 = s0[0]
    # print(f"{s0=}")

    Avg = 70
    sensitivity = 2900
    volts_in_bin = 0.5 / 46.0

    Q = 2 * abs(np.imag(s0) / np.real(s0))
    A = c * np.imag(s0) * np.exp(-1j * 1 / 1)
    maxresp = abs(
        (c / (1j * np.imag(s0) - s0))
        + np.conj(c) / (1j * np.imag(s0) - np.conj(s0))
        + D
        + h * 1j * np.imag(s0)
    )
    displacement = maxresp * sensitivity / (Avg * Q)
    piezomodule = maxresp * sensitivity / (Avg * Q * volts_in_bin)

    mfs = fs
    ms = 1j * mfs

    if doPlot:
        modfun = lambda s: (c / (s - s0) + c / (s - np.conj(s0)) + D + h * s)
        resp = np.array([modfun(si) for si in ms])

        plt.figure(4)

        plt.subplot(3, 2, 1)
        plt.title("Abs")
        plt.plot(fs, abs(data) * 2, "g")
        plt.plot(fs, abs(dfilt) * 2, "k")
        plt.plot(mfs, abs(resp) * 2, "r.")

        # ... Remaining plotting code ...

        plt.show()

    return A, s0, D, h, maxresp, displacement, piezomodule


# Example usage:
# fs and data should be numpy arrays containing frequency and response data respectively.
# A, s0, D, h, maxresp, displacement, piezomodule = vfit(fs, data, doPlot=True)
