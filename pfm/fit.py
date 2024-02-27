"""
Module providing functionality for fitting PFM data.

This module includes functions to fit data obtained from BE PFM
experiments. The goal is to extract meaningful parameters that can be
used for further analysis of the ferroelectric material properties.
"""

import logging
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.fft import fft  # type: ignore
from scipy.signal import butter, filtfilt  # type: ignore
from tqdm import trange  # type: ignore


def fit_data(
    scan_pfm: NDArray[np.complex64],
    cal_pfm: NDArray[np.complex64],
    metadata: dict,
    fc: float = 0.62e6,
    fspan: float = 195312.5,
    **kwargs: NDArray[np.complex64]
) -> dict[str, NDArray[np.complex64]]:
    r"""Fits PFM data at each point using `_vfit` to obtain response data.

    :param scan_pfm: The PFM scan data.
    :param cal_pfm: The data for PFM calibration.
    :param metadata: Data containing info about the scan parameters.
    :param fc: The center frequency.
    :param fspan: The frequency span.
    :param \**kwargs: Additional keyword arguments for
        the fitting process.

    :return: A dictionary containing the reponse data for
        each point in the scan data.
    """
    logging.info("starting fitting process...")

    sizex, sizey = scan_pfm.shape[:2]

    # Initialization
    keys = [
        "A",
        "f0",
        "Q",
        "D",
        "h",
        "c",
        "AQ",
        "s0",
        "large",
        "maxresp",
        "displacement",
        "piezomodule",
    ]

    results = {key: np.full((sizex, sizey), np.nan, dtype="complex_") for key in keys}

    if metadata["new_version"]:
        HLEN = 126  # Number of frequency bins
    else:
        HLEN = 510

    for nyCurve in trange(sizey, desc="progress"):
        for nxCurve in range(sizex):
            indata = scan_pfm[nxCurve, nyCurve, :]
            indata = fft(indata)
            indata = indata / cal_pfm  # calibration # type: ignore

            resp_pfm = np.flip(np.concatenate((indata[-HLEN - 1 :], indata[1:HLEN])))

            fs = np.linspace(fc - fspan / 2, fc + fspan / 2, resp_pfm.size)
            A, s0, D, h, maxresp, displacement, piezomodule = _vfit(fs, resp_pfm)

            results["A"][nxCurve, nyCurve] = A
            results["f0"][nxCurve, nyCurve] = abs(np.imag(s0))
            results["Q"][nxCurve, nyCurve] = abs(np.imag(s0) / np.real(s0))
            results["D"][nxCurve, nyCurve] = D
            results["h"][nxCurve, nyCurve] = h
            results["maxresp"][nxCurve, nyCurve] = maxresp
            results["displacement"][nxCurve, nyCurve] = displacement
            results["piezomodule"][nxCurve, nyCurve] = piezomodule
            results["s0"][nxCurve, nyCurve] = s0

    logging.info("fitting is done")

    return results


def _vfit(
    fs: NDArray[np.float64], data: NDArray[np.complex64], plot: bool = False
) -> Any:
    """Uses `vector fitting <https://scikit-rf.readthedocs.io/en/latest/tutorials/VectorFitting.html>`_
    algorithm for fitting BE PFM data in a single point

    Also see `this article <https://www.sintef.no/globalassets/project/vectfit/vector_fitting_1999.pdf>`_
    for details

    :param fs: Frequency span.
    :param data: PFM data for single point.
    :param plot: Whether to plot results during fitting.
    :return: Response data for the single point.
    """

    def iter(pole: complex, s: NDArray[np.complex64], data: NDArray[np.complex64]):
        """Performs vector fitting iteration

        # TODO: add descriptions
        :param pole: _description_.
        :param s: _description_.
        :param data: _description_.
        :return: _description_.
        """
        real = 1.0 / (s - pole) + 1.0 / (s - np.conj(pole))
        imag = 1j / (s - pole) - 1j / (s - np.conj(pole))

        lhs = np.zeros((len(fs), 4 + 2), dtype=complex)
        lhs[:, :4] = np.column_stack((real, imag, -data * real, -data * imag))
        lhs[:, -2] = 1
        lhs[:, -1] = s
        lhs = np.vstack((np.real(lhs), np.imag(lhs)))

        rhs = data
        rhs = np.concatenate((np.real(rhs), np.imag(rhs)))
        res = np.linalg.pinv(lhs) @ rhs

        p_real, p_imag = np.real(pole), np.imag(pole)
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
    s0 = [-fc / 100 + 1j * fc]

    niters = 30
    fb, fa = butter(8, 0.1)
    dfilt = filtfilt(fb, fa, data, padlen=3 * (max(len(fb), len(fa)) - 1))

    for _ in range(niters):
        s0, c, D, h = iter(pole=s0, s=s, data=dfilt)
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

    if plot:
        mfs = fs
        ms = 1j * mfs

        modfun = lambda s: (c / (s - s0) + c / (s - np.conj(s0)) + D + h * s)
        resp = np.array([modfun(si) for si in ms])

        plt.figure(4)

        plt.subplot(3, 2, 1)
        plt.title("Abs")
        plt.plot(fs, abs(data) * 2, "g")
        plt.plot(fs, abs(dfilt) * 2, "k")
        plt.plot(mfs, abs(resp) * 2, "r.")

        plt.subplot(3, 2, 2)
        plt.plot(fs, np.angle(data), "c", label="data")
        plt.plot(fs, np.angle(dfilt), "k", label="dfilt")
        plt.plot(mfs, np.unwrap(np.angle(resp)), "r.", label="resp")
        plt.legend()

        plt.subplot(3, 2, 5)
        plt.title("Abs(Resp - D - s*h)")
        plt.plot(fs, np.abs(data - D - ms * h) * 2, "g", label="data")
        plt.plot(fs, np.abs(dfilt - D - ms * h) * 2, "k", label="dfilt")
        plt.plot(fs, np.abs(resp - D - ms * h) * 2, "r.", label="resp")
        plt.legend()

        plt.subplot(3, 2, 6)
        plt.title("angle(Resp - D - s*h)")
        plt.plot(fs, np.angle(data - D - ms * h), "c", label="data")
        plt.plot(fs, np.angle(dfilt - D - ms * h), "k", label="dfilt")
        plt.plot(fs, np.angle(resp - D - ms * h), "r.", label="resp")
        plt.legend()

        plt.subplot(3, 2, 3)
        plt.plot(mfs, np.real(resp), "m", label="resp")
        plt.plot(fs, np.real(data), "b", label="data")
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(mfs, np.imag(resp), "m", label="resp")
        plt.plot(fs, np.imag(data), "b", label="data")
        plt.legend()

        delay = 0.5
        plt.ion()  # enable interactive plotting
        plt.pause(delay)  # pause between plots
        plt.clf()  # clear previous results
        plt.show()

    return A, s0, D, h, maxresp, displacement, piezomodule
