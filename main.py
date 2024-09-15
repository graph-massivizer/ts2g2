from core.model import TimeseriesArrayStream
from core.model_previous import Timeseries
from generation.strategies import RandomWalkWithRestartSequenceGenerationStrategy
from timeseries.strategies import TimeseriesToGraphStrategy, TimeseriesEdgeVisibilityConstraintsHorizontal, TimeseriesEdgeVisibilityConstraintsNatural, EdgeWeightingStrategyNull
import networkx as nx
import matplotlib.pyplot as plt

def main():
    #stream = TimeseriesArrayStream([2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3])
    stream = TimeseriesArrayStream([0.35, 0.9, 0.3, 0.8, 0.65, 0.95, 0.1, 0.4, 0.9, 0.2, 0.5, 1.0, 0.05, 0.1, 0.25, 0.75, 0.65, 0.9, 0.4, 0.5])
    timeseries = Timeseries(stream)
    ts2g = TimeseriesToGraphStrategy([TimeseriesEdgeVisibilityConstraintsHorizontal()], "undirected", EdgeWeightingStrategyNull())
    g = ts2g.to_graph(stream)

    nx.draw(g.graph)
    plt.show()

    sequence = g.to_sequence(RandomWalkWithRestartSequenceGenerationStrategy(), sequence_length=500)
    print('Original sequence: {}'.format(timeseries.to_sequence()))
    print('Generated sequence: {}'.format(sequence))

if __name__ == "__main__":
    main()