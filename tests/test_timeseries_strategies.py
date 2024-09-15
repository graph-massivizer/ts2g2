import unittest
from unittest.mock import Mock

from core import model
from core.model import TimeseriesArrayStream
from timeseries.strategies import TimeseriesGraph
from timeseries.strategies import TimeseriesToGraphStrategy

class TestToGraphMethod(unittest.TestCase):

    def test_to_graph(self):
        class MockVisibilityConstraint:
            def is_obstructed(self, timeseries, x1, x2, y1, y2):
                return False  # Mock always returns False for visibility

        # Set up the test
        self.strategy_instance = TimeseriesToGraphStrategy([MockVisibilityConstraint()])

        # Call the to_graph method with the mock timeseries stream

        stream = model.TimeseriesArrayStream([1, 2, 3, 4, 5])


        result_graph = self.strategy_instance.to_graph(stream)

        # Check if the result is an instance of TimeseriesGraph
        self.assertIsInstance(result_graph, TimeseriesGraph)

        # Check if the graph has been constructed correctly
        expected_nodes = [0, 1, 2, 3, 4]
        expected_edges = [(0, 1, {}), (1, 2, {}),
                          (2, 3, {}), (3, 4, {})]
        self.assertEqual(sorted(result_graph.graph.nodes()), expected_nodes)
        self.assertEqual(sorted(result_graph.graph.edges(data=True)), expected_edges)
    def test_to_graph_with_empty_array(self):
        class MockVisibilityConstraint:
            def is_obstructed(self, timeseries, x1, x2, y1, y2):
                return False  # Mock always returns False for visibility


        # Set up the test
        self.strategy_instance = TimeseriesToGraphStrategy([MockVisibilityConstraint()])

        stream = model.TimeseriesArrayStream([])

        result_graph = self.strategy_instance.to_graph(stream)
        print(result_graph.graph)


        # Check if the result is an instance of TimeseriesGraph
        self.assertIsInstance(result_graph, TimeseriesGraph)


        # Check if the graph has been constructed correctly
        expected_nodes = [0, 1, 2, 3, 4]
        expected_edges = [(0, 1, {}), (1, 2, {}),
                          (2, 3, {}), (3, 4, {})]

        self.assertNotEqual(sorted(result_graph.graph.nodes()), expected_nodes)
        self.assertNotEqual(sorted(result_graph.graph.edges(data=True)), expected_edges)




    class TestListElements(unittest.TestCase):
        class MockVisibilityConstraint:
            def is_int_or_float(value):
                return isinstance(value, (int, float))
            def is_obstructed(self, timeseries, x1, x2, y1, y2):
                return False  # Mock always returns False for visibility

            # Set up the test

        self.strategy_instance = TimeseriesToGraphStrategy([MockVisibilityConstraint()])

        # Call the to_graph method with the mock timeseries stream

        stream = model.TimeseriesArrayStream([1, 2, 3, 4, 5])

        result_graph = self.strategy_instance.to_graph(stream)

        def test_integers_and_floats(self):
            my_list = [1, 2.5, 3, 4.7]
            for value in my_list:
                self.assertTrue(self.is_int_or_float(value), f"Value {value} is not an int or float")

        def test_strings_and_characters(self):
            my_list = [1, 'a',3, "world"]
            for value in my_list:
                self.assertFalse(self.is_int_or_float(value), f"Value {value} should not be an int or float")

if __name__ == '__main__':
    unittest.main()

