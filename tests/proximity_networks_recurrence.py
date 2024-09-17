import unittest
import numpy as np
import networkx as nx
from timeseries import proximity_networks


class TestRecurrenceNetwork(unittest.TestCase):
    def setUp(self):
        self.time_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.k = 3
        self.epsilon = 1.0

    def test_epsilon_recurrence_network_basic(self):
        # Test the ε-Recurrence Network with a simple time series and epsilon
        recurrence_network = proximity_networks.ProximityNetwork.RecurrenceNetwork(self.time_series, self.k, self.epsilon)
        G = recurrence_network.create(recurrence_type="epsilon")

        # Debug output
        print(f"Nodes in epsilon recurrence network: {G.nodes()}")
        print(f"Edges in epsilon recurrence network: {G.edges(data=True)}")

        # Check that the correct nodes are created
        self.assertEqual(len(G.nodes), len(self.time_series))

        # Check edges based on epsilon threshold
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            self.assertLessEqual(weight, self.epsilon,
                                 f"Edge weight for edge ({u}, {v}) should be below or equal to epsilon")

    def test_epsilon_recurrence_network_no_edges(self):
        very_small_epsilon = 0.5

        recurrence_network = proximity_networks.ProximityNetwork.RecurrenceNetwork(self.time_series, self.k, very_small_epsilon)
        G = recurrence_network.create(recurrence_type="epsilon")
        print(len(G.edges))

        # Check that there are no edges
        self.assertEqual(len(G.edges), 0, "Expected no edges in the graph")

    def test_knn_network_basic(self):
        # Test the k-Nearest Neighbor Network (k-NNN) with a simple time series
        recurrence_network = proximity_networks.ProximityNetwork.RecurrenceNetwork(self.time_series, self.k, self.epsilon)
        G = recurrence_network.create(recurrence_type="k-nnn")


        # Check that the correct nodes are created
        self.assertEqual(len(G.nodes), len(self.time_series))

        # Check that each node is connected to exactly k neighbors
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            self.assertTrue(len(neighbors) <= self.k, f"Node {node} should have at most {self.k} neighbors")

    def test_knn_network_insufficient_neighbors(self):
        # Test with k larger than the number of nodes
        large_k = len(self.time_series) + 5
        recurrence_network = proximity_networks.ProximityNetwork.RecurrenceNetwork(self.time_series, large_k, self.epsilon)
        G = recurrence_network.create(recurrence_type="k-nnn")

        # Check that the graph is connected correctly
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            self.assertTrue(len(neighbors) == len(self.time_series) - 1,
                            f"Node {node} should be connected to all other nodes")

    def test_adaptive_knn_network_basic(self):
        # Test the Adaptive k-Nearest Neighbor Network (ANNN) with a simple time series
        recurrence_network = proximity_networks.ProximityNetwork.RecurrenceNetwork(self.time_series, self.k, self.epsilon)
        G = recurrence_network.create(recurrence_type="annn")

        # Debug output
        print(f"Nodes in adaptive k-NNN network: {G.nodes()}")
        print(f"Edges in adaptive k-NNN network: {G.edges(data=True)}")

        # Check that the correct nodes are created
        self.assertEqual(len(G.nodes), len(self.time_series))

        # Check that each node is connected based on adaptive threshold
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            self.assertLess(weight, np.max(self.time_series) * 1.2,
                            f"Edge weight for edge ({u}, {v}) should be below adaptive threshold")

    def test_adaptive_knn_network_density_effect(self):
        # Test with time series having repeated values to check adaptive behavior
        time_series = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        recurrence_network = proximity_networks.ProximityNetwork.RecurrenceNetwork(time_series, self.k, self.epsilon)
        G = recurrence_network.create(recurrence_type="annn")

        # Debug output
        print(f"Nodes in adaptive k-NNN network with repeated values: {G.nodes()}")
        print(f"Edges in adaptive k-NNN network with repeated values: {G.edges(data=True)}")

        # Check that the correct nodes are created
        self.assertEqual(len(G.nodes), len(time_series))

        # Check that each node is connected to neighbors based on local density
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            self.assertLess(weight, np.max(time_series) * 1.2,
                            f"Edge weight for edge ({u}, {v}) should be below adaptive threshold")


if __name__ == '__main__':
    unittest.main()