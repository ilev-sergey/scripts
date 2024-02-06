import logging
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from pfm.fitScanPFM import fitScanPFM
from pfm.reader import get_data, load_results


def process_all_data(
    data_folder: Path,
    results_folder: Path | str,
    functions: list[Callable],
    cache: bool = False,
):
    datafiles = data_folder.glob("**/*.nc")
    for datafile in datafiles:
        results_subfolder = Path(
            results_folder,
            *(datafile.relative_to(data_folder).parts[:-1]),
            datafile.stem,
        )
        if cache and (results_subfolder / "results.npy").exists():
            results = load_results(results_subfolder / "results.npy")
        else:
            data = get_data(datafile)
            results = fitScanPFM(**data)
        for function in functions:
            function(results, results_subfolder)


def flip_results(results_filename: Path):
    results = np.load(results_filename, allow_pickle=True).item()
    for key in results.keys():
        results[key] = results[key][::-1]
    np.save(path, results)
    logging.info(f"results mirrored, path: {results_filename}")


def get_domains_distribution(input_folder: Path):
    def calculate_domains(phase: NDArray[np.float64]):
        max = np.quantile(phase, 0.95)
        min = np.quantile(phase, 0.05)
        diff = max - min
        if diff < np.pi / 2:
            return 0, phase.mean()
        threshold = (max + min) / 2
        small_phase = np.count_nonzero(phase <= threshold)
        share = 1 - small_phase / phase.size

        return share, threshold

    pathlist = input_folder.glob("**/results.npy")
    shares = []
    for file in pathlist:
        phase = np.angle(np.load(file, allow_pickle=True).item()["A"])
        share, _threshold = calculate_domains(phase)
        shares.append(share)

    return np.array(shares)


def copy_to_root(root_path: Path, name="phase.png"):
    pathlist = root_path.glob(f"**/{name}")
    for path in pathlist:
        file_path = Path(
            root_path,
            "|".join((path.relative_to(root_path).parts[:-1])) + " " + path.name,
        )
        shutil.copy2(path, file_path)


def save_results(results: dict, output_folder: Path | str):
    output_folder = Path(output_folder)
    Path.mkdir(output_folder, parents=True, exist_ok=True)
    np.save(output_folder / "results.npy", results)  # type: ignore
    logging.info(f"data with fitting results is saved,  path: {output_folder}")


if __name__ == "__main__":
    for path in Path().glob("results/#7755_2/"):
        shares = get_domains_distribution(path)
        shares = np.delete(shares, 2)
        cycles = [0, 0.5, 1.5, 3.5, 5.5, 10.5, 20.5]
        plt.plot(cycles, shares, label=path.stem)
        plt.legend()

        plt.savefig("domains")
