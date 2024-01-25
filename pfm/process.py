import logging
from pathlib import Path

import numpy as np

from pfm.fitScanPFM import fitScanPFM
from pfm.plotter import plot_amplitude, plot_params, plot_phase, plot_piezo
from pfm.reader import get_data


def process_all_data(data_path: Path, results_path: Path, functions):
    pathlist = data_path.glob("**/*.nc")
    for file in pathlist:
        path = Path(results_path, Path(*file.parts[1:-1]) / file.stem)
        data = get_data(file)
        results = fitScanPFM(**data)
        for function in functions:
            function(results, path)


def save_results(results, path: Path):  # TODO: move to another file
    Path.mkdir(path, parents=True, exist_ok=True)

    phase = np.angle(results["A"] * np.exp(-1j * np.pi / 10))
    abs = np.abs(results["A"])
    piezomodule = np.abs(results["piezomodule"])

    # TODO: add check if files exist doesn't process data
    np.savetxt(path / "phase.txt", phase)  # TODO: simplify
    np.save(path / "phase.npy", phase)
    np.savetxt(path / "abs.txt", abs)
    np.save(path / "abs.npy", abs)
    np.savetxt(path / "piezo.txt", piezomodule)
    np.save(path / "piezo.npy", piezomodule)

    logging.info("data with fitting results is saved")


def is_processed(results_path):  # TODO: pass datafile, not results path
    return Path(results_path, "phase.npy").exists()


if __name__ == "__main__":
    # TODO: make that both results or data from files can be passed into main processing functions
    print(is_processed("results/#7752_2/data_2023_12_06__04_56_51 +2.6"))
