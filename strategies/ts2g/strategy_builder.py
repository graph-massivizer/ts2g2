__all__ = [
    "BuildStrategyForTimeseriesToGraph",
    "BuildTimeseriesToGraphNaturalVisibilityStrategy",
    "BuildTimeseriesToGraphHorizontalVisibilityStrategy",
    "BuildTimeseriesToGraphOrdinalPartition",
    "BuildTimeseriesToGraphQuantile",
    "BuildTimeseriesToGraphProximityNetwork",
    "BuildTimeseriesToGraphPearsonCorrelation",
    "TS2G_STRATEGY_REGISTRY"
]
            


from .ordinal_pattern import *
from .proximity import *
from .quantile import *
from .single_node_hash import *
from .visibility import *

#visibility graph strategy
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
        return VisibilityGraphStrategy(
            visibility_constraints = self.visibility,
            graph_type= self.graph_type, 
            edge_weighting_strategy=self.edge_weighting_strategy 
        )


#natural visibility strategy builder
class BuildTimeseriesToGraphNaturalVisibilityStrategy(BuildStrategyForTimeseriesToGraph):
    """As initial strategy sets Natural visibility strategy."""
    def __init__(self):
        super().__init__()
        self.visibility = [TimeseriesEdgeVisibilityConstraintsNatural()]

    def with_limit(self, limit): #实现了limit方法，实现了重写
        self.visibility[0] = TimeseriesEdgeVisibilityConstraintsNatural(limit)
        return self

#horizontal visibility strategy builder
class BuildTimeseriesToGraphHorizontalVisibilityStrategy(BuildStrategyForTimeseriesToGraph):
    """As initial strategy sets Horizontal visibility strategy."""
    def __init__(self):
        super().__init__()
        self.visibility = [TimeseriesEdgeVisibilityConstraintsHorizontal()]

    def with_limit(self, limit): 
        self.visibility[0] = TimeseriesEdgeVisibilityConstraintsHorizontal(limit)
        return self

#ordinal
class BuildTimeseriesToGraphOrdinalPartition:

    def __init__(self, w, tau, use_quantiles=False, Q=4):
        self.w = w
        self.tau = tau
        self.use_quantiles = use_quantiles
        self.Q = Q
    
    def get_strategy(self):
        """Returns strategy."""
        return OrdinalPatternGraphStrategy(self.w, self.tau, self.use_quantiles, self.Q)

#quantile
class BuildTimeseriesToGraphQuantile:
    
    def __init__(self, Q, phi):
        self.Q = Q
        self.phi = phi

    def get_strategy(self):
        return QuantileTransitionGraphStrategy(self.Q, phi = self.phi)

#proximity
class BuildTimeseriesToGraphProximityNetwork:

    def __init__(self, method="cycle", segment_length=10, threshold=0.5, k=5, epsilon=0.5, recurrence_type="epsilon"):
        self.method = method
        self.segment_length = segment_length
        self.treshold = threshold
        self.k = k
        self.epsilon = epsilon
        self.recurrence_type = recurrence_type
    
    def get_strategy(self):
        return ProximityGraphStrategy(method=self.method, segment_length=self.segment_length, threshold=self.treshold, k=self.k, epsilon=self.epsilon, recurrence_type=self.recurrence_type)
    
#
class BuildTimeseriesToGraphPearsonCorrelation:
    def __init__(self):
        pass

    def get_strategy(self):
        return SingleNodeHashGraphStrategy()
"""Registry mapping strategy names to their builder classes for lookup by name."""
# Strategy builder registry for lookup by name
TS2G_STRATEGY_REGISTRY = {
    "natural": BuildTimeseriesToGraphNaturalVisibilityStrategy,
    "horizontal": BuildTimeseriesToGraphHorizontalVisibilityStrategy,
    "ordinal": BuildTimeseriesToGraphOrdinalPartition,
    "quantile": BuildTimeseriesToGraphQuantile,
    "proximity": BuildTimeseriesToGraphProximityNetwork,
    "pearson": BuildTimeseriesToGraphPearsonCorrelation,
}