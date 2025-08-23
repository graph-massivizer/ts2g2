import os, sys
sys.path.append("/var/scratch/sdu/ts2g2")

import numpy as np
from pipeline import T2G2Pipeline
from visualization import draw_g, draw_sequences
from strategies.ts2g.strategy_builder import get_ts2g_strategy_builder
from strategies.view import get_view_strategy
from strategies.g2ts import (
    get_g2ts_strategy,
    StrategySelectNextNodeRandomWithRestart,
    StrategyNextValueInNodeRoundRobin,
)

def main():
    arr = np.array([0.35, 0.9, 0.3, 0.8, 0.65, 0.95, 0.1, 0.4, 0.9, 0.2,
                    0.5, 1.0, 0.05, 0.1, 0.25, 0.75, 0.65, 0.9, 0.4, 0.5], dtype=float)

    view_strategy = get_view_strategy("slice")(start_index=0, end_index=len(arr))
    hv_builder = get_ts2g_strategy_builder(method="horizontal_visibility")
    ts2g_hv = hv_builder().get_strategy()

    pipeline = (
        T2G2Pipeline()
        .load_timeseries(arr)
        .viewing(view_strategy)
        .to_graph(ts2g_hv)
    )

    G = pipeline.get_Graph()
    draw_g(G)

    g2ts_cls = get_g2ts_strategy("plain")
    g2ts = (
        g2ts_cls()
        .next_node_strategy(StrategySelectNextNodeRandomWithRestart())
        .next_value_strategy(StrategyNextValueInNodeRoundRobin())
        .ts_length(len(arr))
    )

    sequences = pipeline.to_timeseries(g2ts).sequences
    seq_gen = sequences[0] if isinstance(sequences, (list, tuple)) else sequences

    print("Original sequence:", arr.tolist())
    print("Generated sequence:", list(seq_gen))

    draw_sequences([arr, np.asarray(seq_gen)],
                   node_strategy_name="RandomRestart",
                   value_strategy_name="RoundRobin")

if __name__ == "__main__":
    main()