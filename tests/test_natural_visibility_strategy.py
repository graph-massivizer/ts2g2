import unittest

from to_graph.strategies import TimeseriesEdgeVisibilityConstraintsNatural


class TestTimeseriesEdgeVisibilityConstraintsNatural(unittest.TestCase):

    def setUp(self):
        self.constraint = TimeseriesEdgeVisibilityConstraintsNatural()

    def test_no_obstruction_increase(self):
        # No obstruction: simple increasing timeseries
        timeseries = [1, 2, 3, 4, 5]
        x1, x2 = 0, 4
        y1, y2 = timeseries[x1], timeseries[x2]
        self.assertFalse(self.constraint.is_obstructed(timeseries, x1, x2, y1, y2))

    def test_no_obstruction_decrease(self):
        # No obstruction: simple increasing timeseries
        timeseries = [5, 4, 3, 2, 1]
        x1, x2 = 0, 4
        y1, y2 = timeseries[x1], timeseries[x2]
        self.assertFalse(self.constraint.is_obstructed(timeseries, x1, x2, y1, y2))

    def test_obstruction(self):
        # Obstruction: a peak in the middle
        timeseries = [1, 2, 10, 4, 5]
        x1, x2 = 0, 4
        y1, y2 = timeseries[x1], timeseries[x2]
        self.assertTrue(self.constraint.is_obstructed(timeseries, x1, x2, y1, y2))

    def test_edge_case_no_obstruction(self):
        # Edge case: same value at all points
        timeseries = [3, 3, 3, 3, 3]
        x1, x2 = 0, 4
        y1, y2 = timeseries[x1], timeseries[x2]
        self.assertFalse(self.constraint.is_obstructed(timeseries, x1, x2, y1, y2))

    def test_edge_case_with_limit(self):
        # Test with limit: Obstruction is allowed within the limit
        self.constraint.limit = 1
        timeseries = [1, 2, 5, 4, 5]
        x1, x2 = 0, 4
        y1, y2 = timeseries[x1], timeseries[x2]
        self.assertTrue(self.constraint.is_obstructed(timeseries, x1, x2, y1, y2))

    def test_edge_case_with_limit_no_obstruction(self):
        # Test with limit: No obstruction within the limit
        self.constraint.limit = 1
        timeseries = [1, 2, 2, 4, 5]
        x1, x2 = 0, 4
        y1, y2 = timeseries[x1], timeseries[x2]
        self.assertFalse(self.constraint.is_obstructed(timeseries, x1, x2, y1, y2))

if __name__ == '__main__':
    unittest.main()
