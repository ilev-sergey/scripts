"""
Module for reading PFM data files (.nc).

It provides functionality to extract the data into numpy arrays for
further processing and analysis.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import netCDF4  # type: ignore
import numpy as np
from numpy.typing import NDArray
from tqdm import trange  # type: ignore


def get_data(
    data_filename: Union[Path, str], print_params: bool = False
) -> Dict[str, Union[np.ndarray, Dict[str, bool]]]:
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
    afam = dataset.groups["data_afam"]

    new_version = "data_freq" in dataset.groups.keys()
    if new_version:
        freq = dataset.groups["data_freq"]

    calibrations_pfm = calibrations.variables["pfm"][:]
    real_calibrations_pfm = np.array(calibrations_pfm[:, 0], dtype=np.float64)
    imag_calibrations_pfm = np.array(calibrations_pfm[:, 1], dtype=np.float64)
    calibrations_pfm = real_calibrations_pfm + 1j * imag_calibrations_pfm

    calibrations_afam = calibrations.variables["afam"][:]
    real_calibrations_afam = np.array(calibrations_afam[:, 0], dtype=np.float64)
    imag_calibrations_afam = np.array(calibrations_afam[:, 1], dtype=np.float64)
    calibrations_afam = real_calibrations_afam + 1j * imag_calibrations_afam

    rows = pfm.variables["waveform"].shape[0] - 1
    cols = pfm.variables["waveform"].shape[1]

    scan_pfm = []
    scan_afam = []
    frequencies = []
    for row in trange(rows, desc="progress"):
        pfmcol = []
        afamcol: List[complex] = []
        freqcol = []
        for col in range(cols):
            datareal_pfm = pfm.variables["waveform"][row, col, :, 0]
            dataimag_pfm = pfm.variables["waveform"][row, col, :, 1]
            # datareal_afam = afam.variables["waveform"][row, col, :, 0]
            # dataimag_afam = afam.variables["waveform"][row, col, :, 1]
            data_pfm = datareal_pfm + 1j * dataimag_pfm
            # data_afam = datareal_afam + 1j * dataimag_afam
            pfmcol.append(data_pfm)
            # afamcol.append(data_afam)

            if new_version:
                data_freqs = freq.variables["waveform"][row, col]  # type: ignore
                freqcol.append(data_freqs)

        scan_pfm.append(pfmcol)
        scan_afam.append(afamcol)
        frequencies.append(freqcol)
    dataset.close()

    logging.info("data is loaded")

    return {
        "scan_pfm": np.array(scan_pfm),
        "scan_afam": np.array(scan_afam),
        "cal_pfm": np.array(calibrations_pfm),
        "cal_afam": np.array(calibrations_afam),
        "frequencies": np.array(frequencies),
        "metadata": {"new_version": new_version},  # | vars(dataset),
    }


def load_results(
    results_filename: Union[Path, str]
) -> Dict[str, NDArray[np.complex64]]:
    """Loads cached fitting results from the file.

    :param results_filename: Path to the cached results file.
    :return: Dictionary containing scan data.
    """
    results = np.load(results_filename, allow_pickle=True).item()
    logging.info(f"loaded cached results from {results_filename}")
    return results


def parse_filename(
    filename: Union[Path, str],
) -> Dict[str, Union[str, datetime, re.Match, None]]:
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
