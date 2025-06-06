__all__ = ["get_view_strategy"]

import numpy as np
from strategies.register import register_strategy
from strategies.register import get_strategy
from core import Timeseries

class BaseViewStrategy:
    def apply(self, time_series: Timeseries):
        raise NotImplementedError("Must be implemented by subclass.")

@register_strategy("view", "sliding_window")
class FixedOverlapWindow(BaseViewStrategy):
    def __init__(self, window_size=100, step_size=10):
        self.window_size = window_size
        self.step_size = step_size

    def apply(self, time_series: Timeseries):
        n = len(time_series)
        windows = []
        for start in range(0, n - self.window_size + 1, self.step_size):
            end = start + self.window_size
            windows.append(time_series[start:end])
        return windows


@register_strategy("view", "non_overlap_window")
class NonOverlapWindow(BaseViewStrategy):
    def __init__(self, window_size=100):
        self.window_size = window_size

    def apply(self, time_series: Timeseries):
        n = len(time_series)
        windows = []
        for start in range(0, n - self.window_size + 1, self.window_size):
            end = start + self.window_size
            windows.append(time_series[start:end])
        return windows



# Example default size function
def default_size_function(start_index, time_series):
    # Always return a fixed window size of 100
    return 100

@register_strategy("view", "dynamic_window")
class DynamicWindow(BaseViewStrategy):
    def __init__(self, size_function=default_size_function):
        self.size_function = size_function

    def apply(self, time_series: Timeseries):
        n = len(time_series)
        windows = []
        start = 0
        while start < n:
            current_size = self.size_function(start, time_series)
            end = min(start + current_size, n)
            windows.append(time_series[start:end])
            start = end
        return windows


# SliceView
@register_strategy("view", "slice")
class SliceView(BaseViewStrategy):
    def __init__(self, start_index, end_index):
        self.start_index = start_index
        self.end_index = end_index

    def apply(self, time_series: Timeseries):
        return [time_series[self.start_index:self.end_index]]

# Get window strategy by name
def get_view_strategy(name: str, **kwargs):
    cls = get_strategy("view", name)
    return cls(**kwargs)