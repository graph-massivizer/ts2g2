import unittest
import numpy as np
import networkx as nx
# from timeseries.quantiles import TimeseriesToQuantileGraph
from to_graph import TimeseriesToQuantileGraph


class TestTimeseriesToQuantileGraph(unittest.TestCase):

    def setUp(self):
        self.Q = 4  # Number of quantiles
        self.phi = 1
        self.ts_to_qg = TimeseriesToQuantileGraph(self.Q, self.phi)

    def test_discretize_to_quantiles(self):
        time_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        quantile_bins, quantile_indices = self.ts_to_qg.discretize_to_quantiles(time_series)

        # Expected quantile bins for 4 quantiles including min and max
        expected_quantile_bins = np.array([1.0, 3.25, 5.5, 7.75, 10.0])
        expected_quantile_indices = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3])

        np.testing.assert_almost_equal(quantile_bins, expected_quantile_bins, decimal=8)
        np.testing.assert_array_equal(quantile_indices, expected_quantile_indices)

    def test_to_graph(self):
        time_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        G = self.ts_to_qg.to_graph(time_series)

        # Check nodes and edges in the graph
        self.assertEqual(len(G.nodes), self.Q)
        self.assertEqual(len(G.edges), 7)  # Expected number of edges based on discretization

        # Check edge weights and normalization
        for u, v, d in G.edges(data=True):
            self.assertIn(u, G.nodes)
            self.assertIn(v, G.nodes)
            self.assertIn('weight', d)
            self.assertGreaterEqual(d['weight'], 0.0)
            self.assertLessEqual(d['weight'], 1.0)

        # Check normalization of edge weights
        for node in G.nodes:
            total_weight = sum([G[node][succ]['weight'] for succ in G.successors(node)])
            self.assertAlmostEqual(total_weight, 1.0, places=8)

    def test_single_value_time_series(self):
        time_series = np.array([1])
        quantile_bins, quantile_indices = self.ts_to_qg.discretize_to_quantiles(time_series)

        # With only one value, all indices should be the same
        expected_quantile_bins = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        expected_quantile_indices = np.array([0])

        np.testing.assert_almost_equal(quantile_bins, expected_quantile_bins, decimal=8)
        np.testing.assert_array_equal(quantile_indices, expected_quantile_indices)

        G = self.ts_to_qg.to_graph(time_series)

        self.assertEqual(len(G.nodes), self.Q)
        self.assertEqual(len(G.edges), 0)

    def test_constant_time_series(self):
        time_series = np.array([5, 5, 5, 5])
        quantile_bins, quantile_indices = self.ts_to_qg.discretize_to_quantiles(time_series)

        # With constant values, all indices should be the same
        expected_quantile_bins = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        expected_quantile_indices = np.array([0, 0, 0, 0])

        np.testing.assert_almost_equal(quantile_bins, expected_quantile_bins, decimal=8)
        np.testing.assert_array_equal(quantile_indices, expected_quantile_indices)

        G = self.ts_to_qg.to_graph(time_series, phi = 1 )

        self.assertEqual(len(G.nodes), self.Q)
        self.assertEqual(len(G.edges), 1)

    def test_increasing_time_series(self):
        time_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        G = self.ts_to_qg.to_graph(time_series)

        # All edges should follow the increasing order
        expected_edges = {
            (0, 0): 2/3,
            (0, 1): 1/3,
            (1, 1): 1/2,
            (1, 2): 1/2,
            (2, 2): 1/2,
            (2, 3): 1/2,
            (3, 3): 1.0
        }

        for (u, v, d) in G.edges(data=True):
            self.assertAlmostEqual(d['weight'], expected_edges[(u, v)], places=8)

    def test_to_graph_phi_3(self):
        time_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.ts_to_qg.phi = 3
        G = self.ts_to_qg.to_graph(time_series, phi=3)

        self.assertEqual(len(G.nodes), self.Q)

        expected_edges = {
            (0, 1): 2/3,
            (0, 2): 1/3,
            (1, 2): 1/2,
            (1, 3): 1/2,
            (2, 3): 1
        }

        # Check edges in the graph
        for (u, v, d) in G.edges(data=True):
            self.assertAlmostEqual(d['weight'], expected_edges[(u, v)], places=8)

        # Ensure the correct number of edges
        self.assertEqual(len(G.edges), len(expected_edges))

if __name__ == '__main__':
    unittest.main()
