import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt  # type: ignore


def vfit(fs: NDArray, data: NDArray, doPlot: bool = False):
    def iter(pguess, s: NDArray, data: NDArray):
        pguess = (
            [pguess]
            if type(pguess) == np.complex128 or type(pguess) == np.float64
            else pguess
        )

        lhs = np.zeros((len(fs), 4 * len(pguess) + 2), dtype=complex)
        for i, pole in enumerate(pguess):
            real = 1.0 / (s - pole) + 1.0 / (s - np.conj(pole))
            imag = 1j / (s - pole) - 1j / (s - np.conj(pole))

            lhs[:, 4 * i : 4 * i + 4] = np.column_stack(
                (real, imag, -data * real, -data * imag)
            )
        lhs[:, -2] = 1
        lhs[:, -1] = s
        lhs = np.vstack((np.real(lhs), np.imag(lhs)))

        rhs = data
        rhs = np.concatenate((np.real(rhs), np.imag(rhs)))
        res = np.linalg.pinv(lhs) @ rhs

        p_real, p_imag = np.real(pguess), np.imag(pguess)
        arr1 = np.array([[p_real, p_imag], [-p_imag, p_real]]).reshape(2, 2)
        arr2 = np.array([[2, 0]]).reshape(2, 1)
        arr3 = res[2:4].reshape(1, 2)
        rm = arr1 - arr2 @ arr3

        poles = np.linalg.eigvals(rm)
        cs = res[0] + 1j * res[1]
        d = res[-2]
        h = res[-1]

        return poles, cs, d, h

    s = 1j * fs
    fc = np.mean(fs)
    pguess = [-fc / 100 + 1j * fc]
    s0 = pguess

    niters = 30
    fb, fa = butter(8, 0.1)
    dfilt = filtfilt(fb, fa, data, padlen=3 * (max(len(fb), len(fa)) - 1))

    for _ in range(niters):
        s0, c, D, h = iter(pguess=s0, s=s, data=dfilt)
        s0 = s0[0]

    Avg = 70
    sensitivity = 2900
    volts_in_bin = 0.5 / 46.0

    Q = 2 * abs(np.imag(s0) / np.real(s0))
    A = c * np.imag(s0) * np.exp(-1j)
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
