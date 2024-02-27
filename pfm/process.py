import logging
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from pfm.fit import fit_data
from pfm.read import get_data, load_results


def process_all_data(
    data_folder: Path,
    results_folder: Path | str,
    functions: list[Callable],
    cache: bool = False,
) -> None:
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
            results = fit_data(**data)
        for function in functions:
            function(results, results_subfolder)


def flip_results(results_filename: Path) -> None:
    results = np.load(results_filename, allow_pickle=True).item()
    for key in results.keys():
        results[key] = results[key][::-1]
    np.save(path, results)
    logging.info(f"results mirrored, path: {results_filename}")


def transform_phase(phase: NDArray[np.float64]) -> NDArray[np.float64]:
    a = -2.8 * 180 / np.pi
    b = 2.40 * 180 / np.pi
    c = 90
    d = -180

    k_1 = (d - c) / (b - a)
    k_2 = (b - a) / (d - c)
    m_1 = (c + d - k_1 * (a + b)) / 2
    m_2 = (a + b - k_2 * (c + d)) / 2

    phase *= 180 / np.pi
    for i in range(phase.shape[0]):
        for j in range(phase.shape[1]):
            if a < phase[i, j] < b:
                phase[i, j] = (90 * np.tanh((k_1 * phase[i, j] + m_1) / 5)) * k_2 + m_2
    phase *= np.pi / 180

    return phase


def get_domains_distribution(input_folder: Path) -> dict[str, NDArray[np.float64]]:
    def calculate_domains(phase: NDArray[np.float64]) -> float:
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
    pathlist = root_path.glob(f"**/{name}")
    for path in pathlist:
        file_path = Path(
            root_path,
            "|".join((path.relative_to(root_path).parts[:-1])) + " " + path.name,
        )
        shutil.copy2(path, file_path)


def save_results(results: dict, output_folder: Path | str) -> None:
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
