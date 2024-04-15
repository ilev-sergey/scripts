import logging
from pathlib import Path

import pfm.pool
from pfm.plot import plot_amp_phase, plot_amp_phase_log, plot_params, plot_piezo
from pfm.process import Cache, delete_pictures, process_all_data, save_results

logging.basicConfig(level=logging.DEBUG, format="%(message)s")
# to remove debug messages from matplotlib
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("multiprocessing").setLevel(logging.WARNING)


def main():
    data_path = Path("data")
    results_path = Path("results")
    functions_to_apply = [
        delete_pictures,
        plot_params,
        plot_piezo,
        plot_amp_phase,
        plot_amp_phase_log,
        save_results,
    ]
    process_all_data(data_path, results_path, functions_to_apply, cache=Cache.SKIP)  # type: ignore


if __name__ == "__main__":
    pfm.pool.init()
    main()
