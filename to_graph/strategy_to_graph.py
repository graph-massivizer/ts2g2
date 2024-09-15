from timeseries.strategies import EdgeWeightingStrategyNull, TimeseriesEdgeVisibilityConstraintsHorizontal, TimeseriesEdgeVisibilityConstraintsNatural, TimeseriesEdgeVisibilityConstraintsVisibilityAngle, TimeseriesToGraphStrategy


#TODO: this is a builder!
class BuildStrategyForTimeseriesToGraph:
    """
    Sets and returns a strategy with which we can convert timeseries into graphs.
    
    **Attributes:**

    - `visibility`: an array of visibility constraints strategies
    
    """

    def __init__(self):
        self.visibility = []
        self.graph_type = "undirected"
        self.edge_weighting_strategy = EdgeWeightingStrategyNull()

    def with_angle(self, angle):
        """Sets an angle in which range must a node be to be considered for connection."""
        self.visibility.append(TimeseriesEdgeVisibilityConstraintsVisibilityAngle(angle))
        return self

    def with_limit(self, limit):
        """Sets a limit as to how many data instances two nodes must be apart to be considered for connection."""
        pass

    def get_strategy(self):
        """Returns strategy."""
        return TimeseriesToGraphStrategy(
            visibility_constraints = self.visibility,
            graph_type= self.graph_type,
            edge_weighting_strategy=self.edge_weighting_strategy
        )

class BuildTimeseriesToGraphNaturalVisibilityStrategy(BuildStrategyForTimeseriesToGraph):
    """As initial strategy sets Natural visibility strategy."""
    def __init__(self):
        super().__init__()
        self.visibility = [TimeseriesEdgeVisibilityConstraintsNatural()]

    def with_limit(self, limit):
        self.visibility[0] = TimeseriesEdgeVisibilityConstraintsNatural(limit)
        return self


class BuildTimeseriesToGraphHorizontalVisibilityStrategy(BuildStrategyForTimeseriesToGraph):
    """As initial strategy sets Horizontal visibility strategy."""
    def __init__(self):
        super().__init__()
        self.visibility = [TimeseriesEdgeVisibilityConstraintsHorizontal()]

    def with_limit(self, limit):
        self.visibility[0] = TimeseriesEdgeVisibilityConstraintsHorizontal(limit)
        return self