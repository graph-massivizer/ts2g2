import unittest

from to_graph.strategies import TimeseriesEdgeVisibilityConstraintsHorizontal


class TestTimeseriesEdgeVisibilityConstraintsHorizontal(unittest.TestCase):

    def setUp(self):
        self.constraint = TimeseriesEdgeVisibilityConstraintsHorizontal()

    def test_no_obstruction(self):
        # No obstruction: intermediate points are below max(y1, y2)
        timeseries = [1, 2, 1, 2, 1]
        x1, x2 = 0, 4
        y1, y2 = timeseries[x1], timeseries[x2]
        self.assertFalse(self.constraint.is_obstructed(timeseries, x1, x2, y1, y2))

    def test_obstruction(self):
        # Obstruction: intermediate point is above or equal to max(y1, y2)
        timeseries = [1, 2, 3, 2, 1]
        x1, x2 = 0, 4
        y1, y2 = timeseries[x1], timeseries[x2]
        self.assertTrue(self.constraint.is_obstructed(timeseries, x1, x2, y1, y2))

    def test_edge_case_no_obstruction(self):
        # Edge case: all values are the same and below max(y1, y2)
        timeseries = [2, 2, 2, 2, 2]
        x1, x2 = 0, 4
        y1, y2 = timeseries[x1], timeseries[x2]
        self.assertFalse(self.constraint.is_obstructed(timeseries, x1, x2, y1, y2))

    def test_edge_case_with_limit(self):
        # Test with limit: Obstruction within the limit is ignored
        self.constraint.limit = 1
        timeseries = [1, 2, 3, 2, 1]
        x1, x2 = 0, 4
        y1, y2 = timeseries[x1], timeseries[x2]
        # The point at index 1 should be ignored
        self.assertTrue(self.constraint.is_obstructed(timeseries, x1, x2, y1, y2))

    def test_edge_case_with_limit_no_obstruction(self):
        # Test with limit: No obstruction within the limit
        self.constraint.limit = 1
        timeseries = [1, 2, 2, 2, 1]
        x1, x2 = 0, 4
        y1, y2 = timeseries[x1], timeseries[x2]
        # The point at index 1 should be ignored
        self.assertFalse(self.constraint.is_obstructed(timeseries, x1, x2, y1, y2))


if __name__ == '__main__':
    unittest.main()
