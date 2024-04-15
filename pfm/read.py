"""
Module for reading PFM data files (.nc).

It provides functionality to extract the data into numpy arrays for
further processing and analysis.
"""

import logging
import re
from collections.abc import KeysView
from datetime import datetime, timedelta
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
    SECOND_HARMONIC = "2nd harm"
    """
    Version with second harmonic detection.
    """


def get_mode(dataset: netCDF4.Dataset) -> Mode:
    data_keys = dataset.groups.keys()

    if "data_freq" in data_keys:
        # check second harmonic
        data = dataset.groups["data_pfm"].variables["waveform"]
        calibrations = dataset.groups["calibrations"].variables["pfm"]
        spectrum = np.abs(np.fft.fft(data[0, 0, :, 0]) / calibrations[:, 0])[
            :127
        ]  # spectrum of response at first point
        if (
            spectrum.max() / np.quantile(spectrum, 0.99) > 4
        ):  # if amplitude at one frequency is much greater than the rest, it is probably second harmonic, ~2-3 for basic BE PFM
            return Mode.SECOND_HARMONIC

        return Mode.AFAM_EHNANCED

    if "data_pfm_lf_tors" in data_keys:
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
        rows = scan_data.shape[0]
        cols = scan_data.shape[1]
        bins = scan_data.shape[2]

        real_calibrations = np.array(scan_cal[:, 0], dtype=np.float64)
        imag_calibrations = np.array(scan_cal[:, 1], dtype=np.float64)
        calibrations = real_calibrations + 1j * imag_calibrations

        scan = np.zeros((rows, cols, bins), dtype=np.complex64)
        for row in trange(rows, desc="progress"):
            for col in range(cols):
                data_real = scan_data[row, col, :, 0]
                data_imag = scan_data[row, col, :, 1]
                scan[row, col] = data_real + 1j * data_imag
        return {
            "data": np.array(scan),
            "calibration_data": calibrations,
            "software_version": software_version,
        }

    logging.info(f"loading data from {data_filename}")
    dataset = netCDF4.Dataset(data_filename, "r", format="NETCDF4")

    cols = dataset.groups["data_pfm"].variables["waveform"].shape[1]
    if cols == 0:
        yield {"data": None}

    if print_params:
        with open("parameters" + ".txt", "w") as file:
            file.write(str(dataset) + "\n")
        logging.info("parameters.txt created")

    calibrations = dataset.groups["calibrations"]
    software_version = get_mode(dataset)

    pfm = dataset.groups["data_pfm"].variables["waveform"]
    calibrations_pfm = calibrations.variables["pfm"][:]
    yield {"scan": "PFM"} | get_scan(pfm, calibrations_pfm)

    match software_version:
        case Mode.AFAM_EHNANCED | Mode.SECOND_HARMONIC:
            calibrations_afam = calibrations.variables["afam"][:]
            afam = dataset.groups["data_afam"].variables["waveform"]
            freq = dataset.groups["data_freq"]
            yield {"scan": "AFAM", "frequencies": freq} | get_scan(
                afam, calibrations_afam
            )

        case Mode.DFL_AND_LF:
            pfm_lf = dataset.groups["data_pfm_lf_tors"].variables["waveform"]
            yield {"scan": "PFM LF"} | get_scan(pfm_lf, calibrations_pfm)

    logging.info("data is loaded")


def load_results(results_filename: Path | str) -> dict[str, NDArray[np.complex64]]:
    """Loads cached fitting results from the file.

    :param results_filename: Path to the cached results file.
    :return: Dictionary containing scan data.
    """
    results = np.load(results_filename, allow_pickle=True).item()
    logging.info(f"loaded cached results from {results_filename}")
    return results


def parse_filename(
    filename: Path | str,
) -> dict[str, str | datetime | re.Match | None | float | timedelta]:
    """Extracts scan parameters from the datafile name if possible.

    :param filename: Path to datafile.
    :return: A dictionary containing the parsed elements including
        datetime, comment, voltage, and pulse time.
    """
    filename = Path(filename).stem
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

    voltage_pattern = r"([-+]?\d+(\.\d+)?|fresh)(?: [VvВв])?"
    voltage_match = re.search(voltage_pattern, comment)
    voltage = voltage_match.group(1) if voltage_match else None
    if voltage_match:
        voltage = voltage_match.group(0)
        if voltage != "fresh":
            voltage = float(voltage)

    time_pattern = r"(\d+) ?(mcs|ms|s)"
    pulse_time_match = re.search(time_pattern, comment)
    pulse_time = None
    if pulse_time_match:
        value, unit = pulse_time_match.groups()
        match unit:
            case "mcs":
                pulse_time = timedelta(microseconds=int(value))
            case "ms":
                pulse_time = timedelta(milliseconds=int(value))
            case "s":
                pulse_time = timedelta(seconds=int(value))

    return {
        "datetime": dt_obj,
        "comment": comment,
        "voltage": voltage,
        "pulse_time": pulse_time,
    }
