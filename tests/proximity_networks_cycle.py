import unittest
import numpy as np
import networkx as nx
from timeseries import proximity_networks


class TestCycleNetwork(unittest.TestCase):
    def setUp(self):
        # This method will run before each test
        self.time_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.segment_length = 3
        self.threshold = 0.5
        self.cycle_network = proximity_networks.ProximityNetwork.CycleNetwork(self.time_series, self.segment_length, self.threshold)

    def test_create_cycle_network(self):
        # Create the cycle network
        G = self.cycle_network.create()

        # Debug output
        print(f"Nodes: {G.nodes()}")
        print(f"Edges: {G.edges(data=True)}")

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
        # Create the cycle network with a high threshold
        high_threshold_network = proximity_networks.ProximityNetwork.CycleNetwork(self.time_series, self.segment_length, 1.0)
        G = high_threshold_network.create()

        # Debug output
        print(f"Nodes with high threshold: {G.nodes()}")
        print(f"Edges with high threshold: {G.edges(data=True)}")

        # Check that no edges are created if the threshold is too high
        num_edges = len(G.edges)
        self.assertEqual(num_edges, 0, "No edges should be present if threshold is too high")


if __name__ == '__main__':
    unittest.main()