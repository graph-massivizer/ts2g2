__all__ = ["draw_ts"]

import matplotlib.pyplot as plt
from core import Timeseries

def draw_ts(ts: Timeseries, title="Time Series", x_legend="Time", y_legend="Value", color="blue", save_path=None):
    """Draws a time series plot with given parameters."""
    plt.figure(figsize=(10, 6))
    x = ts.timestamps if ts.timestamps is not None else range(len(ts.get_data()))
    plt.plot(x, ts.data, linestyle='-', color=color)
    plt.title(title)
    plt.xlabel(x_legend)
    plt.ylabel(y_legend)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()