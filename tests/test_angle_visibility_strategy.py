import unittest
import math

from to_graph.strategies import TimeseriesEdgeVisibilityConstraintsVisibilityAngle

class TestTimeseriesEdgeVisibilityConstraintsVisibilityAngle(unittest.TestCase):

    def test_no_obstruction_below_visibility_angle(self):
        constraint = TimeseriesEdgeVisibilityConstraintsVisibilityAngle(visibility_angle=math.pi/4)
        timeseries = [1, 2, 1, 2, 1]
        self.assertFalse(constraint.is_obstructed(timeseries, 0, 1, 1, 2))

    def test_obstruction_above_visibility_angle(self):
        constraint = TimeseriesEdgeVisibilityConstraintsVisibilityAngle(visibility_angle=math.pi/6)
        timeseries = [1, 2, 1, 2, 1]
        self.assertTrue(constraint.is_obstructed(timeseries, 0, 1, 1, 2))

    def test_no_obstruction_with_horizontal_line(self):
        constraint = TimeseriesEdgeVisibilityConstraintsVisibilityAngle(visibility_angle=math.pi/4)
        timeseries = [1, 1, 1, 1, 1]
        self.assertFalse(constraint.is_obstructed(timeseries, 0, 4, 1, 1))

    def test_obstruction_with_horizontal_line(self):
        constraint = TimeseriesEdgeVisibilityConstraintsVisibilityAngle(visibility_angle=0)
        timeseries = [1, 1, 1, 1, 1]
        self.assertTrue(constraint.is_obstructed(timeseries, 0, 4, 1, 1))

    def test_no_obstruction_when_slope_is_inf(self):
        constraint = TimeseriesEdgeVisibilityConstraintsVisibilityAngle(visibility_angle=math.pi/4)
        timeseries = [1, 2, 1, 2, 1]
        self.assertFalse(constraint.is_obstructed(timeseries, 2, 2, 1, 2))

    def test_consider_absolute_value_false(self):
        constraint = TimeseriesEdgeVisibilityConstraintsVisibilityAngle(visibility_angle=-math.pi/4, consider_visibility_angle_absolute_value=False)
        timeseries = [1, 2, 1, 2, 1]
        self.assertTrue(constraint.is_obstructed(timeseries, 0, 1, 1, 2))

    def test_consider_absolute_value_true(self):
        constraint = TimeseriesEdgeVisibilityConstraintsVisibilityAngle(visibility_angle=math.pi/4, consider_visibility_angle_absolute_value=True)
        timeseries = [1, 2, 1, 2, 1]
        self.assertFalse(constraint.is_obstructed(timeseries, 0, 1, 1, 2))

if __name__ == "__main__":
    unittest.main()