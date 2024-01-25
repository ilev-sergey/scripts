import logging
from pathlib import Path

from pfm.fitScanPFM import fitScanPFM
from pfm.plotter import plot_amplitude, plot_params, plot_phase, plot_piezo

# from pfm.plotter import plot_params, plot_piezo
from pfm.process import process_all_data
from pfm.reader import get_data

logging.basicConfig(
    level=logging.INFO, format="%(message)s"
)  # TODO: add times for each step + mb loadbar
# TODO: UI to choose data path and results path
# base_path = "data"
# folder = "piezo"
# filename = "data_2023_11_14__12_38_33 7755_3 3+5cycles -3V.nc"
# path = Path(base_path, folder, filename)
# data = get_data(path)


# results = fitScanPFM(**data)

# base_path = "pics"
# # folder = "piezo"
# # scan_folder = "data_2023_11_14__12_38_33 7755_3 3+5cycles -3V"
# folder = folder
# scan_folder = Path(filename).with_suffix("")

# path = Path(base_path, folder, scan_folder)
# plot_params(results, path)
# plot_piezo(results, path, include_displ=False)

data_path = Path("data")
results_path = Path("results")
functions_to_apply = [
    plot_params,
    plot_piezo,
    plot_phase,
    plot_amplitude,
]
process_all_data(data_path, results_path, functions_to_apply)
