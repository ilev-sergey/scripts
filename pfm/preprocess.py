import logging
from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

from pfm.read import Mode


def preprocess_data(
    data: NDArray[np.complex64],
    calibration_data: NDArray[np.complex64],
    scan: str,
    software_version: Mode,
    metadata: Dict[str, Any],
    **kwargs: NDArray[np.complex64],
):
    if software_version in (Mode.AFAM_ENHANCED, Mode.SECOND_HARMONIC) and scan == "PFM":
        bin_count = 127 * 2  # Number of frequency bins
    else:
        bin_count = 510 * 2

    size_x, size_y = data.shape[:2]
    logging.debug(
        f"size: {size_x}x{size_y}pt, {bin_count=}, {scan=}, {software_version=}"
    )

    data = data[:, :, :, 0] + 1j * data[:, :, :, 1]
    fft_data = np.fft.fft(data)

    calibrations = calibration_data[:, 0] + 1j * calibration_data[:, 1]
    fft_data /= calibrations
    fft_data = np.concatenate(
        (fft_data[:, :, -bin_count // 2 :], fft_data[:, :, : bin_count // 2]), axis=2
    )
    return {
        "data": fft_data,
        "software_version": software_version,
        "metadata": metadata,
        "scan": scan,
        "bin_count": bin_count,
        **kwargs,
    }
