__all__ = ["draw_ts"]

import matplotlib.pyplot as plt
from core import Timeseries
import numpy as np

def draw_ts(ts, title="Time Series", x_legend="Time", y_legend="Value", color="blue", save_path=None):
    """Draws a time series plot with given parameters."""
    plt.figure(figsize=(10, 6))
    if isinstance(ts, Timeseries):
        x = ts.timestamps if ts.timestamps is not None else range(len(ts.get_data()))
        y = ts.get_data()
    elif isinstance(ts, (list, np.ndarray)):
        y = np.array(ts).flatten()
        x = range(len(y))
    else:
        raise TypeError("Input ts must be a Timeseries instance, list, or numpy.ndarray")
    plt.plot(x, y, linestyle='-', color=color)
    plt.title(title)
    plt.xlabel(x_legend)
    plt.ylabel(y_legend)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()