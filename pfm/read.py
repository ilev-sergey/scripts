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
from typing import Generator

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
) -> Generator[dict, None, None]:
    """Extracts data from the specified datafile, yielding it piecemeal
    for each AFM mode.

    :param data_filename: Path to the data file.
    :param print_params: If ``True``, the scan parameters are printed
        to a text file.
    :return: `Generator` containing dictionary with data for each AFM mode.
    """

    def get_scan(scan_data, scan_cal):
        """Helper function for extracting the scan data.

        :param scan_data: Part of the dataset containing the data for specififc AFM mode
        :param scan_cal: Part of the calibration data for specific AFM mode
        :return: Dictionary containing arrays of complex scan data, calibration data and
            software mode used during the measurement
        """
        rows = scan_data.variables["waveform"].shape[0] - 1
        cols = scan_data.variables["waveform"].shape[1] - 1

        real_calibrations = np.array(scan_cal[:, 0], dtype=np.float64)
        imag_calibrations = np.array(scan_cal[:, 1], dtype=np.float64)
        calibrations = real_calibrations + 1j * imag_calibrations

        scan = []
        for row in trange(rows, desc="progress"):
            scan_col = []
            for col in range(cols):
                data_real = scan_data.variables["waveform"][row, col, :, 0]
                data_imag = scan_data.variables["waveform"][row, col, :, 1]
                data = data_real + 1j * data_imag
                scan_col.append(data)
            scan.append(scan_col)
        return {
            "data": np.array(scan),
            "calibration_data": calibrations,
            "software_version": software_version,
        }

    logging.info(f"loading data from {data_filename}")
    dataset = netCDF4.Dataset(data_filename, "r", format="NETCDF4")

    if print_params:
        with open("parameters" + ".txt", "w") as file:
            file.write(str(dataset) + "\n")
        logging.info("parameters.txt created")

    calibrations = dataset.groups["calibrations"]
    software_version = get_mode(dataset.groups.keys())

    pfm = dataset.groups["data_pfm"]
    calibrations_pfm = calibrations.variables["pfm"][:]
    yield {"name": "PFM"} | get_scan(pfm, calibrations_pfm)

    match software_version:
        case Mode.AFAM_EHNANCED:
            calibrations_afam = calibrations.variables["afam"][:]
            afam = dataset.groups["data_afam"]
            freq = dataset.groups["data_freq"]
            yield {"name": "AFAM", "frequencies": freq} | get_scan(
                afam, calibrations_afam
            )

        case Mode.DFL_AND_LF:
            pfm_lf = dataset.groups["data_pfm_lf_tors"]
            yield {"name": "PFM LF"} | get_scan(pfm_lf, calibrations_pfm)

    logging.info("data is loaded")


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
