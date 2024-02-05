import logging
from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from pfm.fitScanPFM import fitScanPFM
from pfm.reader import get_data, load_results


def process_all_data(
    data_path: Path, results_path: Path | str, functions: list[Callable], cache=False
):
    pathlist = data_path.glob("**/*.nc")
    for file in pathlist:
        file_path = Path(
            results_path, *(file.relative_to(data_path).parts[:-1]), file.stem
        )
        if cache and (file_path / "results.npy").exists():
            results = load_results(file_path / "results.npy")
        else:
            data = get_data(file)
            results = fitScanPFM(**data)
        for function in functions:
            function(results, file_path)


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


def save_results(results: dict, path: Path | str):  # TODO: move to another file
    Path.mkdir(path, parents=True, exist_ok=True)
    np.save(path / "results.npy", results)
    logging.info("data with fitting results is saved")


def is_processed(results_path: Path | str):  # TODO: pass datafile, not results path
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
