import numpy as np
from strategies.register import register_strategy, get_strategy

__all__ = ["get_preprocess_strategy"]

@register_strategy("preprocess", "zscore")
def zscore_preprocess(ts):
    from core.time_series import Timeseries
    assert isinstance(ts, Timeseries), "zscore_preprocess expects a Timeseries object"
    data = ts.get_data()
    mean = np.mean(data)
    std = np.std(data) + 1e-8
    new_data = (data - mean) / std
    return Timeseries(
        data=new_data,
        timestamps=ts.timestamps,
        label=ts.label,
        sensor_id=ts.sensor_id,
        source=ts.source,
        sampling_rate=ts.sampling_rate
    )

@register_strategy("preprocess", "minmax")
def minmax_preprocess(ts):
    from core.time_series import Timeseries
    assert isinstance(ts, Timeseries), "minmax_preprocess expects a Timeseries object"
    data = ts.get_data()
    min_val, max_val = np.min(data), np.max(data)
    new_data = (data - min_val) / (max_val - min_val + 1e-8)
    return Timeseries(
        data=new_data,
        timestamps=ts.timestamps,
        label=ts.label,
        sensor_id=ts.sensor_id,
        source=ts.source,
        sampling_rate=ts.sampling_rate
    )

def get_preprocess_strategy(method="zscore"):
    return get_strategy("preprocess", method)