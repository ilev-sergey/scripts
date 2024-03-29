from collections.abc import Sequence

import pandas as pd


class CV:
    def __init__(self, metadata: dict, dataframes: Sequence[pd.DataFrame]) -> None:
        self.data = dataframes[0]
        self.metadata = metadata
        self._init_metadata()

    def _init_metadata(self):
        self.measurement = self.metadata["Measurement Number"]
        self.measurement_id = self.metadata["Measurement ID"]
        self.series_id = self.metadata["SeriesID"]
        self.mode = self.metadata["MeasureMode"]
        self.first_bias = self.metadata["Start"]
        self.second_bias = self.metadata["Stop"]
        self.step = self.metadata["Step"]
        self.sweep_mode = self.metadata["Sweep mode"]
        self.frequency = self.metadata["Frequency"]
        self.steps = self.metadata["RealMeasuredPoints"]
