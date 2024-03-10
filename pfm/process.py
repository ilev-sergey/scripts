"""
Module providing functionality for processing response data.

Works with response data that is received after fitting PFM data
using `pfm.fit` module.
"""

import logging
import shutil
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from pfm.fit import fit_data
from pfm.read import get_data, load_results


class Cache(Enum):
    """
    Enum used to specify how to handle cached results when processing
    data.
    """

    IGNORE = 0
    """
    Ignore cached results and recompute everything.
    """
    USE = 1
    """
    If cached results are found, use them for processing.
    """
    SKIP = 2
    """
    if cached results are found, skip processing this data.
    """


def process_all_data(
    data_folder: Path,
    results_folder: Path | str,
    functions: list[Callable],
    cache: Cache | bool = Cache.SKIP,
) -> None:
    """Convenient interface for processing all data in the given folder.

    :param data_folder: Parent folder with datafiles.
    :param results_folder: Parent folder to save results.
    :param functions: Functions that should be applied
        to processed data.
    :param cache: If ``True`` tries to load cached results from
        **results_folder**. If cache is not found, the data is processed
        in the usual way. Other possible values are `Cache` enum
        members (`Cache.IGNORE` equals to ``False``,
        `Cache.USE` equals to ``True``).
    """
    datafiles = data_folder.glob("**/*.nc")
    for datafile in datafiles:
        results_subfolder = Path(
            results_folder,
            *(datafile.relative_to(data_folder).parts[:-1]),
            datafile.stem,
        )
        cache_enum = Cache(cache)
        match cache_enum:
            case Cache.SKIP if (results_subfolder / "results.npy").exists():
                logging.info(f"skipping {datafile}: results exist")
                continue
            case Cache.USE if (results_subfolder / "results.npy").exists():
                results = load_results(results_subfolder / "results.npy")
            case Cache.IGNORE | _:
                data = get_data(datafile)
                if data["scan_pfm"].shape == (0,):
                    logging.warning(f"skipping {datafile}: empty data")
                    continue
                results = fit_data(**data)
        for function in functions:
            function(results, results_subfolder)


def flip_results(results_filename: Path) -> None:
    """Flip the data in the given results file (in place). As a result,
    maps will be reflected horizontally. Can be used in case of wrong
    choice of direction in Nova during the scanning (right and down
    instead of right and up).

    :param results_filename: The file path to the results.
    """

    results = np.load(results_filename, allow_pickle=True).item()
    for key in results.keys():
        results[key] = results[key][::-1]
    np.save(path, results)
    logging.info(f"results mirrored, path: {results_filename}")


def transform_phase(
    phase: NDArray[np.float64],
    shift: float = -1,
    amplitude: float = np.pi / 2,
    slope: float = 10,
    final_shift: float = 0.3,
) -> NDArray[np.float64]:
    """Transforms the input phase array using `Heaviside step function
    <https://en.wikipedia.org/wiki/Heaviside_step_function#Analytic_approximations>`_.
    Used for better interpretability of phase maps.

    :param phase: Array of phase values.
    :param shift: Vertical shift of step function.
    :param amplitude: Amplitude of step function.
    :param slope: Slope of step function.
    :param final_shift: Used to change colors for tranformed phase.
    :return: An array of transformed phase values.
    """
    return amplitude * (np.tanh(slope * phase) + shift) + final_shift


def get_domains_distribution(input_folder: Path) -> dict[str, NDArray[np.float64]]:
    """Calculates the share of domains with lowest phase in the each
    results file within the input folder. Used to evaluate the wake-up
    dynamics of the sample during PFM measurements.

    :param input_folder: The folder containing files with fitting
        results.
    :return: A dictionary with the distribution of domains.
    """

    def calculate_domains(phase: NDArray[np.float64]) -> float:
        """Helper function, calculates the domains share of the given
        phase array.

        :param phase: Array of phase values.
        :return: The share of pixels with the lowest phase.
        """
        mask = np.logical_and(-np.pi / 2 < phase, phase < np.pi / 2)
        blue_px = phase[mask].size
        share = blue_px / phase.size

        return share

    pathlist = input_folder.glob("**/results.npy")
    shares = []
    for file in pathlist:
        phase = np.angle(np.load(file, allow_pickle=True).item()["A"])
        phase = transform_phase(phase)
        share = calculate_domains(phase)
        shares.append(share)

    return {"blue": np.array(shares)}


def plot_hysteresis(
    voltages: NDArray[np.float64] | list,
    shares: NDArray[np.float64],
    output_folder: Path | str = ".",
    sample: str = "",
    sort: bool = True,
) -> None:
    """Plots hysteresis curve for given voltages and domain shares,
    and saves the plot to the output folder.

    :param voltages: Array of voltages.
    :param shares: Array of shares.
    :param output_folder: Output folder where the plot will be saved.
    :param sample: Sample name to be used in plot label.
    :param sort: If True, sorts the voltages and shares in ascending
        order before plotting.
    """
    output_folder = Path(output_folder)
    if sort:
        voltages, shares = zip(*sorted(zip(voltages, shares)))
    df = pd.DataFrame({"Voltages": voltages, "Shares": shares})

    # correct order of points
    pos = df[df["Voltages"] > 0]
    neg = df[df["Voltages"] < 0]
    df = pd.concat([neg[::-1], pos, neg.iloc[-1::]], ignore_index=True)

    plt.plot(df["Voltages"], df["Shares"], "-o", label=sample)
    plt.xlabel("Pulse voltage, V")
    plt.ylabel("Share of blue domains")
    plt.legend()
    plt.savefig(output_folder / f"hysteresis {(sample)}.png", bbox_inches="tight")


def copy_to_root(root_path: Path, name="phase.png") -> None:
    """Copy files with the given name to the root directory. Used to
    track differences in the maps in more convenient way, without
    having to look in the folders.
    # TODO: maybe change results structure to "results/datafile.npy" +
    "results/phases/ etc.

    :param root_path: The root directory path.
    :param name: The name of the file to be copied.
    """
    pathlist = root_path.glob(f"**/{name}")
    for path in pathlist:
        file_path = Path(
            root_path,
            "|".join((path.relative_to(root_path).parts[:-1])) + " " + path.name,
        )
        shutil.copy2(path, file_path)


def save_results(results: dict, output_folder: Path | str) -> None:
    """Saves fitting results to the specified output folder as a numpy
    file.

    :param results: The fitting results to be saved.
    :param output_folder: The path to the output folder.
    """
    output_folder = Path(output_folder)
    Path.mkdir(output_folder, parents=True, exist_ok=True)
    np.save(output_folder / "results.npy", results)  # type: ignore
    logging.info(f"data with fitting results is saved, path: {output_folder}")


if __name__ == "__main__":
    for path in Path().glob("results/#7755_2/"):
        shares = get_domains_distribution(path)["blue"]
        shares = np.delete(shares, 2)
        cycles = [0, 0.5, 1.5, 3.5, 5.5, 10.5, 20.5]
        plt.plot(cycles, shares, label=path.stem)
        plt.legend()

        plt.savefig("domains")
