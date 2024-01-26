import logging
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

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


def get_domains_distribution(results_path: Path):
    def calculate_domains(phase: NDArray[np.float64]):  # TODO: move to another file
        max = np.quantile(phase, 0.95)
        min = np.quantile(phase, 0.05)
        diff = max - min
        if diff < np.pi / 2:
            return 0, phase.mean()  # TODO: or 1
        threshold = (max + min) / 2
        small_phase = np.count_nonzero(phase <= threshold)
        share = 1 - small_phase / phase.size

        return share, threshold

    pathlist = results_path.glob("**/phase.npy")
    shares = []
    for file in pathlist:
        phase = np.load(file)
        share, _threshold = calculate_domains(phase)
        shares.append(share)

    return np.array(shares)


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
    for path in Path().glob("results/#7755_2/"):
        shares = get_domains_distribution(path)
        shares = np.delete(shares, 2)
        cycles = [0, 0.5, 1.5, 3.5, 5.5, 10.5, 20.5]
        plt.plot(cycles, shares, label=path.stem)
        plt.legend()

        plt.savefig("domains")
