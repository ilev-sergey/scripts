import logging
from pathlib import Path

from pfm.plot import (
    plot_amp_phase_log,
    plot_amplitude,
    plot_params,
    plot_phase,
    plot_piezo,
)
from pfm.process import Cache, delete_pictures, process_all_data, save_results

logging.basicConfig(level=logging.INFO, format="%(message)s")

data_path = Path("data")
results_path = Path("results")
functions_to_apply = [
    delete_pictures,
    plot_params,
    plot_piezo,
    plot_phase,
    plot_amplitude,
    plot_amp_phase_log,
    save_results,
]
process_all_data(data_path, results_path, functions_to_apply, cache=Cache.SKIP)  # type: ignore
