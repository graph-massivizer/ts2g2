import unittest
from unittest.mock import Mock
from strategies.ts2g import VisibilityGraphStrategy
from core import Timeseries, TimeGraph
class MockVisibilityConstraint:       
        def is_obstructed(self, timeseries, x1, x2, y1, y2):
            return False  # Mock always returns False for visibility

class TestToGraphMethod(unittest.TestCase):
    def test_to_graph(self):

        # Set up the test
        self.strategy_instance = VisibilityGraphStrategy([MockVisibilityConstraint()])

        # Call the to_graph method with the mock timeseries stream

        stream = Timeseries([1, 2, 3, 4, 5])


        result_graph = self.strategy_instance.to_graph(stream)

        # Check if the result is an instance of TimeseriesGraph
        self.assertIsInstance(result_graph,TimeGraph)

        # Check if the graph has been constructed correctly
        expected_nodes = [0, 1, 2, 3, 4]
        expected_edges = [(0, 1, {'color': '#ff8000'}), (0, 2, {'color': '#ff8000'}), 
                          (0, 3, {'color': '#ff8000'}), (0, 4, {'color': '#ff8000'}), 
                          (1, 2, {'color': '#ff8000'}), (1, 3, {'color': '#ff8000'}), 
                          (1, 4, {'color': '#ff8000'}), (2, 3, {'color': '#ff8000'}), 
                          (2, 4, {'color': '#ff8000'}), (3, 4, {'color': '#ff8000'})]
        self.assertEqual(sorted(result_graph.graph.nodes()), expected_nodes)
        self.assertEqual(sorted(result_graph.graph.edges(data=True)), expected_edges)
    
    def test_to_graph_with_empty_array(self):

        # Set up the test
        self.strategy_instance = VisibilityGraphStrategy([MockVisibilityConstraint()])

        stream = Timeseries([])

        result_graph = self.strategy_instance.to_graph(stream)
        print(result_graph.graph)


        # Check if the result is an instance of TimeseriesGraph
        self.assertIsInstance(result_graph, TimeGraph)


        # Check if the graph has been constructed correctly
        expected_nodes = [0, 1, 2, 3, 4]
        expected_edges = [(0, 1, {'color': '#ff8000'}), (1, 2, {'color': '#ff8000'}), 
                          (2, 3, {'color': '#ff8000'}), (3, 4, {'color': '#ff8000'})]

        self.assertNotEqual(sorted(result_graph.graph.nodes()), expected_nodes)
        self.assertNotEqual(sorted(result_graph.graph.edges(data=True)), expected_edges)




class TestListElements(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        # Set up the test
        self.strategy_instance = VisibilityGraphStrategy([MockVisibilityConstraint()])

        stream = Timeseries([1, 2, 3, 4, 5])

        result_graph = self.strategy_instance.to_graph(stream)

    def is_int_or_float(self, value):
        return isinstance(value, (int, float))

    def test_integers_and_floats(self):
        my_list = [1, 2.5, 3, 4.7]
        for value in my_list:
            self.assertTrue(self.is_int_or_float(value), f"Value {value} is not an int or float")

    def test_strings_and_characters(self):
        my_list = [1, 'a', 3, "world"]
        result = all(self.is_int_or_float(value) for value in my_list)
        self.assertFalse(result, "Not all values are int or float")

if __name__ == '__main__':
    unittest.main()

