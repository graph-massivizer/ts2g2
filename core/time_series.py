__all__ = ["Timeseries"]
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Timeseries:
    """
    Container for a univariate time series with support for optional metadata.

    Attributes:
        data (np.ndarray): Time series data values.
        timestamps (np.ndarray or None): Optional time axis.
        label (str or None): Optional label for classification.
        sensor_id (str or None): Optional sensor identifier.
        source (str or None): Data source description.
        sampling_rate (float or None): Sampling rate in Hz.
    """

    def __init__(self, data, timestamps=None, label=None, sensor_id=None, source=None, sampling_rate=None, backend="numpy"):
        assert isinstance(data, (list, np.ndarray)), \
            f"Expected list or np.ndarray for data, got {type(data)}"
        self.backend = backend
        self.data = self._init_data(data)
        self.timestamps = np.array(timestamps) if timestamps is not None else None
        self.label = label
        self.sensor_id = sensor_id
        self.source = source
        self.sampling_rate = sampling_rate

    def _init_data(self, data):
        if self.backend == "numpy":
            return np.array(data)
        elif self.backend == "raw":
            return list(data)
        else:
            raise NotImplementedError(f"Unsupported backend: {self.backend}")

    def __repr__(self):
        return f"Timeseries(len={len(self.data)}, backend='{self.backend}', label={self.label})"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            sliced_data = self.data[idx]
            sliced_timestamps = self.timestamps[idx] if self.timestamps is not None else None
            return Timeseries(
                data=sliced_data,
                timestamps=sliced_timestamps,
                label=self.label,
                sensor_id=self.sensor_id,
                source=self.source,
                sampling_rate=self.sampling_rate,
                backend=self.backend
            )
        else:
            return self.data[idx]

    def get_data(self):
        return self.data

    def apply_preprocess(self, method="zscore"):
        """
        Convenience method for applying a preprocessing strategy directly on the Timeseries.
        Prefer using T2G2Pipeline in production code.

        Parameters:
            method (str): Name of the registered preprocessing strategy.

        Returns:
            Timeseries: New Timeseries instance with preprocessed data.
        """
        from strategies.preprocess import get_preprocess_strategy
        func = get_preprocess_strategy(method)
        return Timeseries(
            data=func(self.data),
            timestamps=self.timestamps,
            label=self.label,
            sensor_id=self.sensor_id,
            source=self.source,
            sampling_rate=self.sampling_rate
        )

    def to_numpy(self):
        if self.backend == "numpy":
            return self.data
        elif self.backend == "raw":
            return np.array(self.data)
        else:
            raise NotImplementedError(f"Unsupported backend: {self.backend}")

    def length(self):
        return len(self.data)
    
