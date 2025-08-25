__all__ = ["draw_sequences"]

import matplotlib.pyplot as plt

def plot_timeseries(sequence, title, x_legend="Date", y_legend="Value", color="black"):
    """Draw one timeseries."""
    plt.figure(figsize=(10, 6))
    plt.plot(sequence, linestyle='-', color=color)
    plt.title(title)
    plt.xlabel(x_legend)
    plt.ylabel(y_legend)
    plt.grid(True)
    plt.show()


def draw_sequences(sequences, node_strategy_name, value_strategy_name):
    """Draw multiple timeseries sequences."""
    if not sequences or len(sequences) == 0:
        raise ValueError("No sequences to draw")

    title_prefix = f"next_node_strategy = {node_strategy_name}, next_value = {value_strategy_name}"

    for j, seq in enumerate(sequences):
        color = "black"  
        plot_timeseries(seq, title_prefix, "Date", "Value", color)