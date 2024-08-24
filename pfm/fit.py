"""
Module providing functionality for fitting PFM data.

This module includes functions to fit data obtained from BE PFM
experiments. The goal is to extract meaningful parameters that can be
used for further analysis of the ferroelectric material properties.
"""

import logging
import re
from functools import partial
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.fft import fft  # type: ignore
from scipy.signal import butter, filtfilt  # type: ignore
from tqdm import tqdm  # type: ignore

from pfm.read import Mode


def get_response(s0: complex, c: complex, D: int, h: int, avg_count: int):
    sensitivity = 2900
    volts_in_bin = 0.5 / 46.0

    Q = 2 * abs(np.imag(s0) / np.real(s0))
    A = c * np.imag(s0)
    maxresp = abs(
        (c / (1j * np.imag(s0) - s0))
        + np.conj(c) / (1j * np.imag(s0) - np.conj(s0))
        + D
        + h * 1j * np.imag(s0)
    )
    displacement = maxresp * sensitivity / (avg_count * Q)
    piezomodule = maxresp * sensitivity / (avg_count * Q * volts_in_bin)

    return {
        "amplitude": A,
        "resonant_frequency": abs(np.imag(s0)),
        "Q_factor": abs(np.imag(s0) / np.real(s0)),
        "D": D,
        "h": h,
        "max_response": maxresp,
        "displacement": displacement,
        "piezomodule": piezomodule,
        "s0": s0,
    }


def fit_line(
    line_data: NDArray[np.complex64],
    calibration_data: NDArray[np.complex64],
    bin_count: int,
    central_freq: float,
    freq_span: float,
    avg_count: int,
    software_version: Mode,
    scan: str,
) -> dict[str, NDArray[np.complex64]]:
    """Fits the given line data using `_vfit` to obtain response data.
    Used as a task for `Pool`.
    """
    response_keys = get_response(1 + 1j, 1j, 1, 1, 1).keys()  # dummy call to get keys
    results_line = {
        key: np.full((line_data.shape[0]), np.nan, dtype="complex_")
        for key in response_keys
    }
    fs = np.linspace(
        central_freq - freq_span / 2, central_freq + freq_span / 2, bin_count
    )
    for row in range(line_data.shape[0]):
        data_in_point = line_data[row, :]

        data_to_fit = fft(data_in_point)
        data_to_fit = data_to_fit / calibration_data  # type: ignore
        data_to_fit = np.flip(data_to_fit)
        data_to_fit = np.concatenate(
            (data_to_fit[-bin_count // 2 :], data_to_fit[: bin_count // 2])
        )

        if (
            software_version == Mode.SECOND_HARMONIC and scan == "PFM"
        ):  # skip fitting for second harmonic
            response = {key: 0.0 for key in response_keys}  # fill with zeros
            response["s0"] = 1e-10

            second_harm_bin = np.abs(data_to_fit).argmax()
            response["amplitude"] = data_to_fit[second_harm_bin]

        else:
            fit_results = _vfit(fs, data_to_fit)
            response = get_response(**fit_results, avg_count=avg_count)

        for key in response.keys():
            results_line[key][row] = response[key]

    return results_line


def fit_data(
    data: NDArray[np.complex64],
    calibration_data: NDArray[np.complex64],
    scan: str,
    software_version: Mode,
    metadata: dict[str, Any],
    **kwargs: NDArray[np.complex64],
) -> dict[str, str | dict[str, NDArray[np.complex64]]]:
    r"""Fits the given scan data at each point using `_vfit` to obtain response data.

    :param data: The scan data.
    :param cal_data: The calibration data for given scan.

    :return: A dictionary containing response data for each point in the scan data.
    """
    logging.info(f"starting {scan} data fitting...")

    if software_version in (Mode.AFAM_ENHANCED, Mode.SECOND_HARMONIC) and scan == "PFM":
        bin_count = 127 * 2  # Number of frequency bins
    else:
        bin_count = 510 * 2

    size_x, size_y = data.shape[:2]
    logging.debug(
        f"size: {size_x}x{size_y}pt, {bin_count=}, {scan=}, {software_version=}"
    )

    central_freq = (
        float(re.search(r"(\d+\.?\d*) MHz", metadata["center_freq"]).group(1)) * 1e6
    )
    freq_span = float(re.search(r"(\d+\.?\d*)KHz", metadata["span"]).group(1)) * 1e3
    pfm_avg = int(metadata["pfm_avg"])

    from pfm.pool import pool

    task = partial(
        fit_line,
        calibration_data=calibration_data,
        bin_count=bin_count,
        central_freq=central_freq,
        freq_span=freq_span,
        avg_count=pfm_avg,
        software_version=software_version,
        scan=scan,
    )
    a = list(tqdm(pool.imap(task, data), total=size_x, desc="progress"))
    # transform from list of dicts to dict of arrays
    results = {k: np.array([dic[k] for dic in a]) for k in a[0]}

    logging.info("fitting is done")

    return {"scan": scan, "response_data": results}


def _vfit(
    freq_span: NDArray[np.float64],
    data: NDArray[np.complex64],
    plot: bool = False,
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

    def iter(
        pole: np.complex64,
        s: NDArray[np.complex64],
        data: NDArray[np.complex64],
        n: int = 1,
    ):
        """Performs vector fitting iteration

        # TODO: add descriptions
        :param pole: _description_.
        :param s: _description_.
        :param data: _description_.
        :return: _description_.
        """
        for _ in range(n):
            real = 1.0 / (s - pole) + 1.0 / (s - np.conj(pole))
            imag = 1j / (s - pole) - 1j / (s - np.conj(pole))

            lhs = np.zeros((len(freq_span), 4 + 2), dtype=complex)
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
            pole = poles[0]

        return pole, cs, d, h

    s = 1j * freq_span
    fc = np.mean(freq_span)
    s0 = -fc / 100 + 1j * fc

    niters = 30
    fb, fa = butter(8, 0.1)
    dfilt = filtfilt(fb, fa, data, padlen=3 * (max(len(fb), len(fa)) - 1))

    s0, c, D, h = iter(pole=s0, s=s, data=dfilt, n=niters)

    if plot:
        mfs = freq_span
        ms = 1j * mfs

        modfun = lambda s: (c / (s - s0) + c / (s - np.conj(s0)) + D + h * s)
        resp = np.array([modfun(si) for si in ms])

        plt.figure(4)

        plt.subplot(3, 2, 1)
        plt.title("Abs")
        plt.plot(freq_span, abs(data) * 2, "g")
        plt.plot(freq_span, abs(dfilt) * 2, "k")
        plt.plot(mfs, abs(resp) * 2, "r.")

        plt.subplot(3, 2, 2)
        plt.plot(freq_span, np.angle(data), "c", label="data")
        plt.plot(freq_span, np.angle(dfilt), "k", label="dfilt")
        plt.plot(mfs, np.unwrap(np.angle(resp)), "r.", label="resp")
        plt.legend()

        plt.subplot(3, 2, 5)
        plt.title("Abs(Resp - D - s*h)")
        plt.plot(freq_span, np.abs(data - D - ms * h) * 2, "g", label="data")
        plt.plot(freq_span, np.abs(dfilt - D - ms * h) * 2, "k", label="dfilt")
        plt.plot(freq_span, np.abs(resp - D - ms * h) * 2, "r.", label="resp")
        plt.legend()

        plt.subplot(3, 2, 6)
        plt.title("angle(Resp - D - s*h)")
        plt.plot(freq_span, np.angle(data - D - ms * h), "c", label="data")
        plt.plot(freq_span, np.angle(dfilt - D - ms * h), "k", label="dfilt")
        plt.plot(freq_span, np.angle(resp - D - ms * h), "r.", label="resp")
        plt.legend()

        plt.subplot(3, 2, 3)
        plt.plot(mfs, np.real(resp), "m", label="resp")
        plt.plot(freq_span, np.real(data), "b", label="data")
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(mfs, np.imag(resp), "m", label="resp")
        plt.plot(freq_span, np.imag(data), "b", label="data")
        plt.legend()

        delay = 0.5
        plt.ion()  # enable interactive plotting
        plt.pause(delay)  # pause between plots
        plt.clf()  # clear previous results
        plt.show()

    return {"s0": s0, "c": c, "D": D, "h": h}
