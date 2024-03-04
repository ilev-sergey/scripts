"""
Module providing functionality for plotting response data.

Works with response data that is received after fitting PFM data
using `pfm.fit` module. It includes functions to create 2D maps for
amplitude, phase, and fitting parameters, as well as histograms for
piezoresponse distribution. The module is a wrap around matplotlib
functions designed for the purpose of generation of high-quality figures
with consistent style suitable for both analysis and publication.
"""

from pathlib import Path
from typing import Any, Tuple, Union

import cmocean  # type: ignore
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1  # type: ignore
import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm  # type: ignore

from pfm.process import transform_phase


def _plot_map(
    fig: mpl.figure.Figure,
    ax: mpl.axes.Axes,
    data: NDArray[np.float64],
    title: Union[str, None] = None,
    vmin: Union[float, None] = None,
    vmax: Union[float, None] = None,
    quantiles: Tuple[float, float] = (0.05, 0.95),
    cmap: Any = "gray",
) -> None:
    """Plots a map of the given 2D data on the specified figure and
    axes. Used as as a helper function for other plot functions for
    a consistent image style.

    :param fig: The figure to plot on.
    :param ax: The axes to plot on.
    :param data: The 2D data to be plotted.
    :param title: The title of the plot.
    :param vmin: The minimum value of the color scale.
    :param vmax: The maximum value of the color scale.
    :param quantiles: The quantiles for **vmin** and **vmax** calculation.
        Used only if **vmin** and **vmax** are not specified.
    :param cmap: The colormap to be used for plotting.
    """
    vmin = vmin or float(np.quantile(data, quantiles[0]))
    vmax = vmax or float(np.quantile(data, quantiles[1]))
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


def plot_amp_phase_log(results: dict, output_folder: Path) -> None:
    """Generates a figure with maps of amplitude, phase
    and log of amplitude from fitting results and
    saves it in the specified output folder.

    :param results: Fitting results with data to be plotted.
    :param output_folder: The folder where the plot will be saved.
    """
    Path.mkdir(output_folder, parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    phase = transform_phase(np.angle(results["A"]))

    _plot_map(
        fig,
        axs[1, 0],
        phase,
        title="Phase",
        cmap=cmocean.cm.phase,
        vmin=-np.pi,
        vmax=np.pi,
    )
    abs = np.abs(results["A"])
    _plot_map(fig, axs[0, 0], abs, title="Abs")
    _plot_map(fig, axs[0, 1], np.log10(abs), title=r"$log_{10}$ Abs")

    fig.delaxes(axs[1, 1])
    plt.tight_layout()
    fig.savefig(output_folder / "amp_phase_log.png")
    plt.close(fig)


def plot_amp_phase(results: dict, output_folder: Path) -> None:
    """Generates a figure with maps of amplitude and phase from
    fitting results and saves it in the specified output folder.

    :param results: Fitting results with data to be plotted.
    :param output_folder: The folder where the plot will be saved.
    """
    Path.mkdir(output_folder, parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    phase = transform_phase(np.angle(results["A"]))

    _plot_map(
        fig,
        axs[1],
        phase,
        title="Phase",
        cmap=cmocean.cm.phase,
        vmin=-np.pi,
        vmax=np.pi,
    )
    abs = np.abs(results["A"])
    _plot_map(fig, axs[0], abs, title="Abs")

    plt.tight_layout()
    fig.savefig(output_folder / "amp_phase.png", bbox_inches="tight")
    plt.close(fig)


def plot_phase(results: dict, output_folder: Path, transformed: bool = True) -> None:
    """Generates a figure with map of phase from fitting results and
    saves it in the specified output folder. Optionally also saves
    the transformed phase.

    :param results: Fitting results with data to be plotted.
    :param output_folder: The folder where the plot will be saved.
    :param transformed: Whether to save the transformed phase.
    """

    def save_image(phase: NDArray[np.float64], img_name: str = "phase.png") -> None:
        """Helper function to plot phasemap and save image.

        :param phase: data with phase to be plotted
        :param img_name: filename of the saved image
        """
        fig, ax = plt.subplots()
        _plot_map(
            fig,
            ax,
            phase,
            title="Phase",
            cmap=cmocean.cm.phase,
            vmin=-np.pi,
            vmax=np.pi,
        )
        fig.savefig(output_folder / img_name, bbox_inches="tight")
        plt.close(fig)

    Path.mkdir(output_folder, parents=True, exist_ok=True)

    phase = np.angle(results["A"] * np.exp(-1j * np.pi / 10))
    save_image(phase)

    if transformed:
        save_image(transform_phase(phase), img_name="phase (transformed).png")


def plot_amplitude(results: dict, output_folder: Path) -> None:
    """Generates a figure with map of amplitude from fitting results
    and saves it in the specified output folder.

    :param results: Fitting results with data to be plotted.
    :param output_folder: The folder where the plot will be saved.
    """
    Path.mkdir(output_folder, parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    amplitude = np.abs(results["A"])
    _plot_map(fig, ax, amplitude, title="Amplitude")
    fig.savefig(output_folder / "amplitude.png", bbox_inches="tight")
    plt.close(fig)


def plot_params(results: dict, output_folder: Path) -> None:
    """Generates a figure with maps of amplitude, phase, contact
    frequency, quality factor and vector fitting parameters (D, H)
    from fitting results and saves it in the specified output folder.

    :param results: Fitting results with data to be plotted.
    :param output_folder: The folder where the plot will be saved.
    """
    Path.mkdir(output_folder, parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 14})

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    phase = np.angle(results["A"] * np.exp(0.1 * 1j * np.pi / 2))
    _plot_map(
        fig,
        axs[0, 1],
        phase,
        title="Phase",
        cmap=cmocean.cm.phase,
        vmin=-np.pi,
        vmax=np.pi,
    )

    _plot_map(fig, axs[0, 0], np.abs(results["A"]), title="Abs")
    _plot_map(fig, axs[0, 2], np.abs(results["f0"]), title="Freq")
    _plot_map(fig, axs[1, 0], np.abs(results["Q"]), title="Q", quantiles=(0.1, 0.9))
    _plot_map(fig, axs[1, 1], np.abs(results["D"]), title="D", quantiles=(0.1, 0.9))
    _plot_map(fig, axs[1, 2], np.abs(results["h"]), title="H", quantiles=(0.0, 1.0))

    plt.tight_layout()
    fig.savefig(output_folder / "params.png")
    plt.close(fig)


def plot_piezo(results: dict, output_folder: Path, include_displ: bool = False) -> None:
    """Generates a figure with map of piezomodule and distribution of
    piezomodule from fitting results and saves it in the specified
    output folder. Optionally also plots the displacement map and
    distribution of displacement.

    :param results: Fitting results with data to be plotted.
    :param output_folder: The folder where the plot will be saved.
    :param include_displ: Whether to plot the displacement map.
    """
    Path.mkdir(output_folder, parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 14})

    fig, axs = plt.subplots(1, 2, figsize=(25, 10))

    piezomodule = np.abs(results["piezomodule"])
    _plot_map(fig, axs[0], piezomodule, title="Piezomodule (pm/V)")

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
        _plot_map(fig, axs[1, 0], displacement, title="Low-frequency displacement [pm]")

        mean_displ = round(np.mean(displacement), 3)
        std_displ = round(np.std(displacement), 3)

        axs[1, 1].hist(displacement, bins=25, color="r")
        axs[1, 1].set_title("Distribution of displacement")
        legend_text = f"mean displacement = {mean_displ} pm/V, std = {std_displ} pm/V"
        axs[1, 1].legend([legend_text], loc="upper left")

    plt.tight_layout()
    fig.savefig(output_folder / "piezomodule.png")
    plt.close(fig)
