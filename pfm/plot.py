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
from typing import Any

import cmocean  # type: ignore
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1  # type: ignore
import numpy as np
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from numpy.typing import NDArray
from scipy.stats import norm  # type: ignore

from pfm.process import transform_phase


def _plot_map(
    data: NDArray[np.float64],
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    quantiles: tuple[float, float] = (0.05, 0.95),
    cmap: Any = "grey",
    size_nm: float = 800e-9,
    scalebar: bool = True,
    fig: mpl.figure.Figure | None = None,
    ax: mpl.axes.Axes | None = None,
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

    def plot_scalebar(ax):
        scalebar = AnchoredSizeBar(
            ax.transData,
            size=100e-9,
            label="100 nm",
            loc="lower right",
            size_vertical=25e-9,
            frameon=True,
            color="black",
            pad=0.2,
            borderpad=0.3,
        )
        ax.add_artist(scalebar)

    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    vmin = vmin or float(np.quantile(data, quantiles[0]))
    vmax = vmax or float(np.quantile(data, quantiles[1]))
    image = ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(0, size_nm, 0, size_nm / data.shape[1] * data.shape[0]),
    )
    if title:
        ax.set_title(title)

    if scalebar:
        plot_scalebar(ax)

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

    # colorbar close to plot, same size
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size=0.10, pad=0.05)
    fig.colorbar(image, cax=cax, orientation="horizontal")


def plot_amp_phase_log(
    results: dict, output_folder: Path, interactive: bool = False
) -> None:
    """Generates a figure with maps of amplitude, phase
    and log of amplitude from fitting results and
    saves it in the specified output folder.

    :param results: Fitting results with data to be plotted.
    :param output_folder: The folder where the plot will be saved.
    """
    Path.mkdir(output_folder, parents=True, exist_ok=True)

    xsize, ysize = mpl.rcParams["figure.figsize"]
    fig, axs = plt.subplots(2, 2, figsize=(xsize * 2, ysize * 2))
    phase = transform_phase(np.angle(results["amplitude"]))

    _plot_map(
        phase,
        fig=fig,
        ax=axs[1, 0],
        title="Phase",
        cmap=cmocean.cm.phase,
        vmin=-np.pi,
        vmax=np.pi,
    )
    abs = np.abs(results[key := "amplitude"])
    _plot_map(
        abs,
        fig=fig,
        ax=axs[0, 0],
        title=key.capitalize(),
    )
    _plot_map(
        np.log10(abs),
        fig=fig,
        ax=axs[0, 1],
        title=r"$log_{10}$ Abs",
    )

    fig.delaxes(axs[1, 1])

    if interactive:
        plt.show()
    else:
        fig.savefig(output_folder / "amp_phase_log.png")
        plt.close(fig)


def plot_amp_phase(
    results: dict, output_folder: Path, interactive: bool = False
) -> None:
    """Generates a figure with maps of amplitude and phase from
    fitting results and saves it in the specified output folder.

    :param results: Fitting results with data to be plotted.
    :param output_folder: The folder where the plot will be saved.
    """
    Path.mkdir(output_folder, parents=True, exist_ok=True)

    xsize, ysize = mpl.rcParams["figure.figsize"]
    fig, axs = plt.subplots(2, 1, figsize=(xsize, ysize * 2))
    phase = transform_phase(np.angle(results["amplitude"]))

    _plot_map(
        phase,
        fig=fig,
        ax=axs[1],
        title="Phase",
        cmap=cmocean.cm.phase,
        vmin=-np.pi,
        vmax=np.pi,
    )
    abs = np.abs(results[key := "amplitude"])
    _plot_map(
        abs,
        fig=fig,
        ax=axs[0],
        title=key.capitalize(),
    )

    if interactive:
        plt.show()
    else:
        fig.savefig(output_folder / "amp_phase.png")
        plt.close(fig)


def plot_phase(
    results: dict,
    output_folder: Path,
    transformed: bool = True,
    interactive: bool = False,
) -> None:
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
            phase,
            fig=fig,
            ax=ax,
            title="Phase",
            cmap=cmocean.cm.phase,
            vmin=-np.pi,
            vmax=np.pi,
        )

        if interactive:
            plt.show()
        else:
            fig.savefig(output_folder / img_name)
            plt.close(fig)

    Path.mkdir(output_folder, parents=True, exist_ok=True)

    phase = np.angle(results["amplitude"])
    save_image(phase)

    if transformed:
        save_image(transform_phase(phase), img_name="phase (transformed).png")

def plot_dfl_lf_combinations(
    dfl_phase: NDArray[np.float64],
    lf_phase: NDArray[np.float64],
    output_folder: Path = Path("."),
    plot_all: bool = True,
    interactive: bool = False,
) -> None:
    dfl_phase = transform_phase(dfl_phase)
    lf_phase = transform_phase(lf_phase)

    xsize, ysize = mpl.rcParams["figure.figsize"]
    fig, axs = plt.subplots(2, 2, figsize=(xsize * 2, ysize * 2))

    dfl_pos_lf_pos = ((lf_phase > 0) + (dfl_phase > 0)).astype(int)
    dfl_pos_lf_neg = ((lf_phase < 0) + (dfl_phase > 0)).astype(int)
    dfl_neg_lf_pos = ((lf_phase > 0) + (dfl_phase < 0)).astype(int)
    dfl_neg_lf_neg = ((lf_phase < 0) + (dfl_phase < 0)).astype(int)

    _plot_map(dfl_pos_lf_pos, cmap="binary", fig=plt.gcf(), ax=axs[0, 0])
    _plot_map(dfl_pos_lf_neg, cmap="binary", fig=plt.gcf(), ax=axs[0, 1])
    _plot_map(dfl_neg_lf_pos, cmap="binary", fig=plt.gcf(), ax=axs[1, 0])
    _plot_map(dfl_neg_lf_neg, cmap="binary", fig=plt.gcf(), ax=axs[1, 1])

    axs[0, 0].set_title("DFL+, LF+")
    axs[0, 1].set_title("DFL+, LF-")
    axs[1, 0].set_title("DFL-, LF+")
    axs[1, 1].set_title("DFL-, LF-")

    if interactive:
        plt.show()
    else:
        fig.savefig(output_folder / "phase_combinations.png")
        plt.close(fig)

    if plot_all:
        fig, ax = plt.subplots()
        result = (
            dfl_pos_lf_pos * 10
            + dfl_pos_lf_neg * 5
            + dfl_neg_lf_pos * (-5)
            + dfl_neg_lf_neg * (-10)
        ).astype(np.float64)
        _plot_map(result, fig=plt.gcf(), ax=ax, cmap="rainbow")

        if interactive:
            plt.show()
        else:
            fig.savefig(output_folder / "all_phases_at_once.png")
            plt.close(fig)
        plt.close(fig)


def plot_amplitude(
    results: dict, output_folder: Path, interactive: bool = False
) -> None:
    """Generates a figure with map of amplitude from fitting results
    and saves it in the specified output folder.

    :param results: Fitting results with data to be plotted.
    :param output_folder: The folder where the plot will be saved.
    """
    Path.mkdir(output_folder, parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    amplitude = np.abs(results[key := "amplitude"])
    _plot_map(
        amplitude,
        fig=fig,
        ax=ax,
        title=key.capitalize(),
    )

    if interactive:
        plt.show()
    else:
        fig.savefig(output_folder / f"{key}.png")
        plt.close(fig)


def plot_params(results: dict, output_folder: Path, interactive: bool = False) -> None:
    """Generates a figure with maps of amplitude, phase, contact
    frequency, quality factor and vector fitting parameters (D, H)
    from fitting results and saves it in the specified output folder.

    :param results: Fitting results with data to be plotted.
    :param output_folder: The folder where the plot will be saved.
    """
    Path.mkdir(output_folder, parents=True, exist_ok=True)

    xsize, ysize = mpl.rcParams["figure.figsize"]
    fig, axs = plt.subplots(2, 3, figsize=(xsize * 3, ysize * 2))

    phase = np.angle(results["amplitude"])
    _plot_map(
        phase,
        fig=fig,
        ax=axs[0, 1],
        title="Phase",
        cmap=cmocean.cm.phase,
        vmin=-np.pi,
        vmax=np.pi,
    )

    _plot_map(
        np.abs(results[key := "amplitude"]),
        fig=fig,
        ax=axs[0, 0],
        title=key.capitalize(),
    )
    _plot_map(
        np.abs(results[key := "resonant_frequency"]),
        fig=fig,
        ax=axs[0, 2],
        title=key.capitalize(),
    )
    _plot_map(
        np.abs(results[key := "Q_factor"]),
        fig=fig,
        ax=axs[1, 0],
        title=key,
        quantiles=(0.1, 0.9),
    )
    _plot_map(
        np.abs(results[key := "D"]),
        fig=fig,
        ax=axs[1, 1],
        title=key,
        quantiles=(0.1, 0.9),
    )
    _plot_map(
        np.abs(results[key := "h"]),
        fig=fig,
        ax=axs[1, 2],
        title=key,
        quantiles=(0.0, 1.0),
    )

    if interactive:
        plt.show()
    else:
        fig.savefig(output_folder / "params.png")
        plt.close(fig)


def plot_piezo(
    results: dict,
    output_folder: Path,
    include_displ: bool = False,
    interactive: bool = False,
) -> None:
    """Generates a figure with map of piezomodule and distribution of
    piezomodule from fitting results and saves it in the specified
    output folder. Optionally also plots the displacement map and
    distribution of displacement.

    :param results: Fitting results with data to be plotted.
    :param output_folder: The folder where the plot will be saved.
    :param include_displ: Whether to plot the displacement map.
    """
    Path.mkdir(output_folder, parents=True, exist_ok=True)

    xsize, ysize = mpl.rcParams["figure.figsize"]
    fig, axs = plt.subplots(1, 2, figsize=(xsize * 2, ysize))

    piezomodule = np.abs(results["piezomodule"])
    _plot_map(
        piezomodule,
        fig=fig,
        ax=axs[0],
        title="Piezomodule (pm/V)",
    )

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
        _plot_map(
            displacement,
            fig=fig,
            ax=axs[1, 0],
            title="Low-frequency displacement [pm]",
        )

        mean_displ = round(np.mean(displacement), 3)
        std_displ = round(np.std(displacement), 3)

        axs[1, 1].hist(displacement, bins=25, color="r")
        axs[1, 1].set_title("Distribution of displacement")
        legend_text = f"mean displacement = {mean_displ} pm/V, std = {std_displ} pm/V"
        axs[1, 1].legend([legend_text], loc="upper left")

    if interactive:
        plt.show()
    else:
        fig.savefig(output_folder / "piezomodule.png")
        plt.close(fig)
