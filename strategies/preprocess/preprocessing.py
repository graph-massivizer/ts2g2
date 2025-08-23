import numpy as np
from typing import Tuple
from strategies.register import register_strategy, get_strategy
from .interface import PreprocessStrategy
__all__ = [
    "get_preprocess_strategy",
    "PreprocessStrategy",
    "ZScorePreprocess",
    "MinMaxPreprocess",
]



@register_strategy("preprocess", "zscore")
class ZScorePreprocess(PreprocessStrategy):
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def transform(self, ts):
        from core.time_series import Timeseries
        assert isinstance(ts, Timeseries), "ZScorePreprocess.transform expects a Timeseries object"
        data = ts.get_data()
        mean = np.mean(data)
        std = np.std(data) + self.eps
        new_data = (data - mean) / std
        return Timeseries(
            data=new_data,
            timestamps=ts.timestamps,
            label=ts.label,
            sensor_id=ts.sensor_id,
            source=ts.source,
            sampling_rate=ts.sampling_rate,
        )
        
@register_strategy("preprocess", "minmax")
class MinMaxPreprocess(PreprocessStrategy):
    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0), eps: float = 1e-8):
        lo, hi = feature_range
        assert hi > lo, "feature_range must be (min, max) with max > min"
        self.lo = lo
        self.hi = hi
        self.eps = eps

    def transform(self, ts):
        from core.time_series import Timeseries
        assert isinstance(ts, Timeseries), "MinMaxPreprocess.transform expects a Timeseries object"
        data = ts.get_data()
        min_val, max_val = np.min(data), np.max(data)
        scaled01 = (data - min_val) / (max_val - min_val + self.eps)
        new_data = scaled01 * (self.hi - self.lo) + self.lo
        return Timeseries(
            data=new_data,
            timestamps=ts.timestamps,
            label=ts.label,
            sensor_id=ts.sensor_id,
            source=ts.source,
            sampling_rate=ts.sampling_rate,
        )

def get_preprocess_strategy(method="zscore"):
    return get_strategy("preprocess", method)