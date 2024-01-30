import logging

import numpy as np
from scipy.fft import fft

from pfm.vfit import vfit


def fitScanPFM(scan_pfm, cal_pfm, metadata, fc=0.62e6, fspan=195312.5, **kwargs):
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
    results["data"] = np.full((sizex, sizey, 1022), np.nan, dtype="complex_")

    if metadata["new_version"]:
        HLEN = 126  # Number of frequency bins
    else:
        HLEN = 510

    for nyCurve in range(sizey):
        for nxCurve in range(sizex):
            indata = scan_pfm[nxCurve, nyCurve, :]
            indata = fft(indata)
            indata = indata / cal_pfm  # calibration

            resp_pfm = np.flip(np.concatenate((indata[-HLEN - 1 :], indata[1:HLEN])))

            fs = np.linspace(fc - fspan / 2, fc + fspan / 2, resp_pfm.size)
            A, s0, D, h, maxresp, displacement, piezomodule = vfit(fs, resp_pfm)

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
