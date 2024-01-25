import logging
from datetime import datetime

import netCDF4
import numpy as np


def get_data(path):
    logging.info(f"loading data from {path}")

    dataset = netCDF4.Dataset(path, "r", format="NETCDF4")

    with open("parameters" + ".txt", "w") as file:
        file.write(str(dataset) + "\n")

    logging.info("\tparameters.txt created")

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

    # print(pfm.variables["waveform"].shape)
    # print(afam.variables["waveform"][0, 0, :, 0])

    scan_pfm = []
    scan_afam = []
    frequencies = []
    for row in range(rows):
        pfmcol = []
        afamcol = []
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
        "metadata": {"new_version": new_version},  # parse_filename(path.name) |
    }


def parse_filename(filename: str):
    filename = str(filename)
    filename, ext = filename.split(".")
    temp = filename.split(" ", 1)
    dt, comment = temp, "" if len(temp) == 1 else temp
    (
        _,
        year,
        month,
        day,
        _,
        hours,
        minutes,
        seconds,
    ) = dt.split("_")
    dt = datetime(year, month, day, hours, minutes, seconds)
    return {"datetime": dt, "comment": comment}
