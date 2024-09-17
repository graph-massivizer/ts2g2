import unittest
import numpy as np
import networkx as nx
from timeseries import proximity_networks # Replace with the actual module name


class TestCorrelationNetwork(unittest.TestCase):
    def setUp(self):
        # This method will run before each test
        self.time_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.segment_length = 3
        self.threshold = 0.5
        self.correlation_network = proximity_networks.ProximityNetwork.CorrelationNetwork(self.time_series, self.segment_length,
                                                                       self.threshold)

    def test_create_correlation_network(self):
        # Create the correlation network
        G = self.correlation_network.create()
        # Check the number of nodes
        expected_num_nodes = (len(self.time_series) - self.segment_length + 1)
        self.assertEqual(len(G.nodes), expected_num_nodes, "Number of nodes should be as expected")

        # Check the number of edges
        num_edges = len(G.edges)
        self.assertTrue(num_edges >= 0, "Number of edges should be non-negative")

        # Check that edges have weights and they are above the threshold
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            self.assertGreaterEqual(weight, self.threshold,
                                    f"Edge weight for edge ({u}, {v}) should be above threshold")

    def test_no_edges_below_threshold(self):
        # Create the correlation network with a high threshold
        high_threshold_network = proximity_networks.ProximityNetwork.CorrelationNetwork(self.time_series, self.segment_length, 1.0)
        G = high_threshold_network.create()

        # Check that no edges are created if the threshold is too high
        num_edges = len(G.edges)
        self.assertEqual(num_edges, 0, "No edges should be present if threshold is too high")

    def test_no_edges_with_low_correlation(self):
        # Create a correlation network with a very low threshold
        low_threshold_network = proximity_networks.ProximityNetwork.CorrelationNetwork(self.time_series, self.segment_length, -1.0)
        G = low_threshold_network.create()

        # Check that edges are created even with very low thresholds
        num_edges = len(G.edges)
        self.assertGreater(num_edges, 0, "Edges should be created even with a very low threshold")

    def test_small_time_series(self):
        # Test with a time series smaller than the segment length
        small_time_series = np.array([1, 2])
        small_network = proximity_networks.ProximityNetwork.CorrelationNetwork(small_time_series, 3, self.threshold)
        G = small_network.create()

        # Check that no nodes or edges are created if the time series is too small
        self.assertEqual(len(G.nodes), 0,
                         "No nodes should be present if the time series is smaller than the segment length")
        self.assertEqual(len(G.edges), 0,
                         "No edges should be present if the time series is smaller than the segment length")


if __name__ == '__main__':
    unittest.main()