import logging
from pathlib import Path
from typing import Any

import cmocean  # type: ignore
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1  # type: ignore
import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm  # type: ignore

# TODO: add typehints


def plot_map(
    fig: mpl.figure.Figure,
    ax: mpl.axes.Axes,
    data: NDArray,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    quantiles: tuple[float, float] = (0.05, 0.95),
    cmap: Any = "grey",
):
    vmin = vmin or np.quantile(data, quantiles[0])
    vmax = vmax or np.quantile(data, quantiles[1])
    image = ax.imshow(
        data,
        cmap=cmap,
        origin="lower",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    if title:
        ax.set_title(title)

    # disable ticks and labels
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )

    # 1e5 -> 10^5 on top of colorbar
    formatter = mpl.ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))

    # colorbar close to plot, same size
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    fig.colorbar(image, cax=cax, orientation="horizontal", format=formatter)


def transform_phase(phase: NDArray):
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


def plot_phase(results: dict, path: Path, transformed: bool = True):
    Path.mkdir(path, parents=True, exist_ok=True)

    phase = np.angle(results["A"] * np.exp(-1j * np.pi / 10))

    fig, ax = plt.subplots()
    plot_map(
        fig,
        ax,
        phase,
        title="Phase",
        cmap=cmocean.cm.phase,
        vmin=-np.pi,
        vmax=np.pi,
    )
    fig.savefig(path / "phase.png", bbox_inches="tight")
    plt.close(fig)

    if transformed:
        fig, ax = plt.subplots()
        plot_map(
            fig,
            ax,
            transform_phase(phase),
            title="Phase",
            cmap=cmocean.cm.phase,
            vmin=-np.pi,
            vmax=np.pi,
        )
        fig.savefig(path / "phase (transformed).png", bbox_inches="tight")
        plt.close(fig)


def plot_amplitude(results: dict, path: Path):
    Path.mkdir(path, parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    amplitude = np.abs(results["A"])
    plot_map(fig, ax, amplitude, title="Amplitude")
    fig.savefig(path / "amplitude.png", bbox_inches="tight")
    plt.close(fig)


def plot_params(results: dict, path: Path):
    Path.mkdir(path, parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 14})

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    phase = np.angle(results["A"] * np.exp(0.1 * 1j * np.pi / 2))
    plot_map(
        fig,
        axs[0, 1],
        phase,
        title="Phase",
        cmap=cmocean.cm.phase,
        vmin=-np.pi,
        vmax=np.pi,
    )

    plot_map(fig, axs[0, 0], np.abs(results["A"]), title="Abs")
    plot_map(fig, axs[0, 2], np.abs(results["f0"]), title="Freq")
    plot_map(fig, axs[1, 0], np.abs(results["Q"]), title="Q", quantiles=(0.1, 0.9))
    plot_map(fig, axs[1, 1], np.abs(results["D"]), title="D", quantiles=(0.1, 0.9))
    plot_map(fig, axs[1, 2], np.abs(results["h"]), title="H", quantiles=(0.0, 1.0))

    plt.tight_layout()
    fig.savefig(path / "params.png")
    plt.close(fig)


def plot_piezo(results: dict, path: Path, include_displ: bool = False):
    Path.mkdir(path, parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 14})

    fig, axs = plt.subplots(1, 2, figsize=(25, 10))

    piezomodule = np.abs(results["piezomodule"])
    plot_map(fig, axs[0], piezomodule, title="Piezomodule (pm/V)")

    D33 = piezomodule.flatten()
    mean_D33 = round(np.mean(D33), 3)
    std_D33 = round(np.std(D33), 3)

    axs[1].hist(D33, bins=25, density=True, alpha=0.6, color="b", edgecolor="black")
    axs[1].set_title("Distribution of piezomodule")
    legend_text = f"mean piezomodule = {mean_D33} pm/V, std = {std_D33} pm/V"
    axs[1].legend([legend_text])

    mu, std = norm.fit(D33)
    xmin, xmax = axs[1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    axs[1].plot(x, p, "k", linewidth=2)

    if include_displ:
        displacement = np.abs(results["displacement"])
        plot_map(fig, axs[1, 0], displacement, title="Low-frequency displacement [pm]")

        mean_displ = round(np.mean(displacement), 3)
        std_displ = round(np.std(displacement), 3)

        axs[1, 1].hist(displacement, bins=25, color="r")
        axs[1, 1].set_title("Distribution of displacement")
        legend_text = f"mean displacement = {mean_displ} pm/V, std = {std_displ} pm/V"
        axs[1, 1].legend([legend_text], loc="upper left")

    plt.tight_layout()
    fig.savefig(path / "piezomodule.png")
    plt.close(fig)
