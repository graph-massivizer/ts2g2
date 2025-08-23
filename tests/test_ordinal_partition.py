import unittest
import numpy as np
import networkx as nx
from strategies.ts2g import get_ts2g_strategy_builder


class TestTimeseriesToOrdinalPatternGraph(unittest.TestCase):
    def setUp(self):
        self.w = 3
        self.tau = 1
        self.obj = get_ts2g_strategy_builder("ordinal")(self.w, self.tau).get_strategy()
        # self.obj = TimeseriesToOrdinalPatternGraph(self.w, self.tau)
        self.time_series = np.array([4, 2, 1, 3, 5])
        self.obj_quantiles = get_ts2g_strategy_builder("ordinal")(self.w, self.tau, use_quantiles=True, Q=4).get_strategy()
        # self.obj_quantiles = TimeseriesToOrdinalPatternGraph(self.w, self.tau, use_quantiles=True, Q=4)

    def test_embeddings(self):
        expected_result = np.array([
            [4, 2, 1],
            [2, 1, 3],
            [1, 3, 5]
        ])
        result = self.obj.embeddings(self.time_series)
        np.testing.assert_array_equal(result, expected_result)

    def test_ordinal_pattern(self):
        vector = np.array([4, 2, 1])
        expected_result = (2, 1, 0)
        result = self.obj.ordinal_pattern(vector)
        self.assertEqual(result, expected_result)

        vector = np.array([1, 3, 5])
        expected_result = (0, 1, 2)
        result = self.obj.ordinal_pattern(vector)
        self.assertEqual(result, expected_result)

    def test_ordinal_pattern_with_quantiles(self):
        vector = np.array([4, 2, 1])
        result = self.obj_quantiles.ordinal_pattern(vector)
        # Since the exact result can vary based on the distribution of the data,
        # we ensure that the result is a tuple of the same length as the input vector.
        self.assertEqual(len(result), len(vector))

        vector = np.array([1, 3, 5])
        result = self.obj_quantiles.ordinal_pattern(vector)
        self.assertEqual(len(result), len(vector))

    def test_to_graph(self):
        G = self.obj.to_graph(self.time_series)
        expected_nodes = {
            (2, 1, 0),  # [4, 2, 1]
            (1, 0, 2),  # [2, 1, 3]
            (0, 1, 2)  # [1, 3, 5]
        }
        expected_edges = {
            ((2, 1, 0), (1, 0, 2)),
            ((1, 0, 2), (0, 1, 2))
        }

        self.assertEqual(set(G.get_graph().nodes), expected_nodes)
        self.assertEqual(set(G.get_graph().edges), expected_edges)

        # Check edge weights
        self.assertAlmostEqual(G.get_graph()[(2, 1, 0)][(1, 0, 2)]['weight'], 1 / 3)
        self.assertAlmostEqual(G.get_graph()[(1, 0, 2)][(0, 1, 2)]['weight'], 1 / 3)

    def test_to_graph_with_quantiles(self):
        G = self.obj_quantiles.to_graph(self.time_series)
        # Verify that the graph has been created and contains nodes and edges
        self.assertGreater(len(G.get_graph().nodes), 0)
        self.assertGreater(len(G.get_graph().edges), 0)
        # Verify that nodes have the correct structure (tuple of length w)
        for node in G.get_graph().nodes:
            self.assertEqual(len(node), self.w)
        # Verify that edge weights are correctly set
        for (u, v, data) in G.get_graph().edges(data=True):
            self.assertIn('weight', data)
            self.assertGreaterEqual(data['weight'], 0)
            self.assertLessEqual(data['weight'], 1)


if __name__ == '__main__':
    unittest.main()
