from pathlib import Path

from pfm.fitScanPFM import fitScanPFM
from pfm.plotter import plot_amplitude, plot_params, plot_phase, plot_piezo
from pfm.reader import get_data


def process_all_data(data_path: Path, results_path: Path):
    pathlist = data_path.glob("**/*.nc")
    for file in pathlist:
        path = Path(results_path, Path(*file.parts[1:-1]) / file.stem)
        data = get_data(file)
        results = fitScanPFM(**data)
        plot_params(results, path)
        plot_piezo(results, path, include_displ=False)
        plot_phase(results, path)
        plot_amplitude(results, path)
