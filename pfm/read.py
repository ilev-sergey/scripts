"""
Module for reading PFM data files (.nc).

It provides functionality to extract the data into numpy arrays for
further processing and analysis.
"""

import logging
import re
from collections.abc import KeysView
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import netCDF4  # type: ignore
import numpy as np
from numpy.typing import NDArray
from tqdm import trange  # type: ignore


class Mode(StrEnum):
    """
    Enum used to specify which mode was used during measurements
    for correct data processing.
    """

    BASE = "PFM"
    """
    Default BE PFM version.
    """
    AFAM_EHNANCED = "AFAM"
    """
    Sped-up version that uses AFAM enhanced resonanse tracking.
    """
    DFL_AND_LF = "PFM LF"
    """
    Version with 2 data channels (DFL and LF).
    """


def get_mode(groups: KeysView):
    if "data_freq" in groups:
        return Mode.AFAM_EHNANCED
    if "data_pfm_lf_tors" in groups:
        return Mode.DFL_AND_LF
    return Mode.BASE


def get_data(
    data_filename: Path | str, print_params: bool = False
) -> dict[str, np.ndarray | None | dict[str, Mode]]:
    """Loads data from the specified datafile and returns it in a
    dictionary.

    :param data_filename: Path to the data file.
    :param print_params: If ``True``, the scan parameters are printed
        to a text file.
    :return: Dictionary containing scan data.
    """
    logging.info(f"loading data from {data_filename}")
    dataset = netCDF4.Dataset(data_filename, "r", format="NETCDF4")

    if print_params:
        with open("parameters" + ".txt", "w") as file:
            file.write(str(dataset) + "\n")
        logging.info("parameters.txt created")

    calibrations = dataset.groups["calibrations"]
    pfm = dataset.groups["data_pfm"]

    software_version = get_mode(dataset.groups.keys())
    match software_version:
        case Mode.AFAM_EHNANCED:
            freq = dataset.groups["data_freq"]
            afam = dataset.groups["data_afam"]
        case Mode.DFL_AND_LF:
            pfm_lf = dataset.groups["data_pfm_lf_tors"]

    calibrations_pfm = calibrations.variables["pfm"][:]
    real_calibrations_pfm = np.array(calibrations_pfm[:, 0], dtype=np.float64)
    imag_calibrations_pfm = np.array(calibrations_pfm[:, 1], dtype=np.float64)
    calibrations_pfm = real_calibrations_pfm + 1j * imag_calibrations_pfm

    if software_version == Mode.AFAM_EHNANCED:
        calibrations_afam = calibrations.variables["afam"][:]
        real_calibrations_afam = np.array(calibrations_afam[:, 0], dtype=np.float64)
        imag_calibrations_afam = np.array(calibrations_afam[:, 1], dtype=np.float64)
        calibrations_afam = real_calibrations_afam + 1j * imag_calibrations_afam

    rows = pfm.variables["waveform"].shape[0] - 1
    cols = pfm.variables["waveform"].shape[1] - 1

    scan_pfm = []
    scan_afam = []
    scan_pfm_lf = []
    frequencies = []
    for row in trange(rows, desc="progress"):
        pfmcol = []
        afamcol = []
        pfmcol_lf = []
        freqcol = []
        for col in range(cols):
            datareal_pfm = pfm.variables["waveform"][row, col, :, 0]
            dataimag_pfm = pfm.variables["waveform"][row, col, :, 1]
            data_pfm = datareal_pfm + 1j * dataimag_pfm
            pfmcol.append(data_pfm)

            if software_version == Mode.AFAM_EHNANCED:
                data_freqs = freq.variables["waveform"][row, col]  # type: ignore
                freqcol.append(data_freqs)

                datareal_afam = afam.variables["waveform"][row, col, :, 0]
                dataimag_afam = afam.variables["waveform"][row, col, :, 1]
                data_afam = datareal_afam + 1j * dataimag_afam
                afamcol.append(data_afam)

            if software_version == Mode.DFL_AND_LF:
                datareal_pfm_lf = pfm_lf.variables["waveform"][row, col, :, 0]
                dataimag_pfm_lf = pfm_lf.variables["waveform"][row, col, :, 1]
                data_pfm_lf = datareal_pfm_lf + 1j * dataimag_pfm_lf
                pfmcol_lf.append(data_pfm_lf)

        scan_pfm.append(pfmcol)
        scan_afam.append(afamcol)
        scan_pfm_lf.append(pfmcol_lf)
        frequencies.append(freqcol)
    dataset.close()

    logging.info("data is loaded")

    return {
        "scan_pfm": np.array(scan_pfm),
        "cal_pfm": np.array(calibrations_pfm),
        "scan_afam": np.array(scan_afam),
        "cal_afam": (
            np.array(calibrations_afam)
            if software_version == Mode.AFAM_EHNANCED
            else None
        ),
        "scan_pfm_lf": np.array(scan_pfm_lf),
        "frequencies": np.array(frequencies),
        "metadata": {"software_version": software_version},
    }


def load_results(results_filename: Path | str) -> dict[str, NDArray[np.complex64]]:
    """Loads cached fitting results from the file.

    :param results_filename: Path to the cached results file.
    :return: Dictionary containing scan data.
    """
    results = np.load(results_filename, allow_pickle=True).item()
    logging.info(f"loaded cached results from {results_filename}")
    return results


def parse_filename(filename: Path | str) -> dict[str, str | datetime | re.Match | None]:
    """Extracts scan parameters from the datafile name if possible.

    :param filename: Path to datafile.
    :return: A dictionary containing the parsed elements including
        datetime, comment, voltage, and pulse time.
    """
    filename = str(filename)
    filename, ext = filename.rsplit(".", 1)
    lst = filename.split(" ", 1)
    if len(lst) == 1:
        dt, comment = *lst, ""
    else:
        dt, comment = lst
    (
        _,
        year,
        month,
        day,
        _,
        hours,
        minutes,
        seconds,
    ) = [int(elem) if elem.isdigit() else elem for elem in dt.split("_")]
    dt_obj = datetime(year, month, day, hours, minutes, seconds)
    voltage_pattern = r"[-+]?\d+(\.\d+)?( [VvВв])?|fresh"
    time_pattern = r"\d+(?:mcs|ms|s)"
    voltage = re.search(voltage_pattern, comment)
    pulse_time = re.search(time_pattern, comment)
    return {
        "datetime": dt_obj,
        "comment": comment,
        "voltage": voltage,
        "pulse_time": pulse_time,
    }
