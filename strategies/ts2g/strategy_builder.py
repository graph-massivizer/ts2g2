__all__ = [
    "BuildVisiblityStrategy",
    "BuildNaturalVisibilityStrategy",
    "BuildHorizontalVisibilityStrategy",
    "BuildOrdinalPartitionStrategy",
    "BuildQuantileStrategy",
    "BuildProximityNetworkStrategy",
    "BuildPearsonCorrelationStrategy",
    "get_ts2g_strategy_builder"
]
            


from .ordinal_pattern import *
from .proximity import *
from .quantile import *
from .single_node_hash import *
from .visibility import *
from .interface import StrategyBuilder, TimeseriesToGraphStrategy
from strategies.register import register_strategy, get_strategy

@register_strategy("ts2g_builder", "visibility")
class BuildVisiblityStrategy(StrategyBuilder):
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

    def get_strategy(self) -> TimeseriesToGraphStrategy:  # Added type annotation
        """Returns strategy.""" 
        return VisibilityGraphStrategy(
            visibility_constraints=self.visibility,
            graph_type=self.graph_type, 
            edge_weighting_strategy=self.edge_weighting_strategy 
        )


@register_strategy("ts2g_builder", "natural_visibility")
class BuildNaturalVisibilityStrategy(BuildVisiblityStrategy):
    """As initial strategy sets Natural visibility strategy."""
    def __init__(self):
        super().__init__()
        self.visibility = [TimeseriesEdgeVisibilityConstraintsNatural()]

    def with_limit(self, limit): 
        self.visibility[0] = TimeseriesEdgeVisibilityConstraintsNatural(limit)
        return self


@register_strategy("ts2g_builder", "horizontal_visibility")
class BuildHorizontalVisibilityStrategy(BuildVisiblityStrategy):
    """As initial strategy sets Horizontal visibility strategy."""
    def __init__(self):
        super().__init__()
        self.visibility = [TimeseriesEdgeVisibilityConstraintsHorizontal()]

    def with_limit(self, limit): 
        self.visibility[0] = TimeseriesEdgeVisibilityConstraintsHorizontal(limit)
        return self


@register_strategy("ts2g_builder", "ordinal")
class BuildOrdinalPartitionStrategy(StrategyBuilder):

    def __init__(self, w, tau, use_quantiles=False, Q=4):
        self.w = w
        self.tau = tau
        self.use_quantiles = use_quantiles
        self.Q = Q
    
    def get_strategy(self) -> TimeseriesToGraphStrategy:  # Added type annotation
        """Returns strategy."""
        return OrdinalPatternGraphStrategy(self.w, self.tau, self.use_quantiles, self.Q)


@register_strategy("ts2g_builder", "quantile")
class BuildQuantileStrategy(StrategyBuilder):
    
    def __init__(self, Q, phi):
        self.Q = Q
        self.phi = phi

    def get_strategy(self) -> TimeseriesToGraphStrategy:  # Added type annotation
        return QuantileTransitionGraphStrategy(self.Q, phi=self.phi)


@register_strategy("ts2g_builder", "proximity")
class BuildProximityNetworkStrategy(StrategyBuilder):

    def __init__(self, method="cycle", segment_length=10, threshold=0.5, k=5, epsilon=0.5, recurrence_type="epsilon"):
        self.method = method
        self.segment_length = segment_length
        self.threshold = threshold
        self.k = k
        self.epsilon = epsilon
        self.recurrence_type = recurrence_type
    
    def get_strategy(self) -> TimeseriesToGraphStrategy:  # Added type annotation
        return ProximityGraphStrategy(
            method=self.method,
            segment_length=self.segment_length,
            threshold=self.threshold,
            k=self.k,
            epsilon=self.epsilon,
            recurrence_type=self.recurrence_type
        )


@register_strategy("ts2g_builder", "pearson")
class BuildPearsonCorrelationStrategy(StrategyBuilder):
    def __init__(self):
        pass

    def get_strategy(self) -> TimeseriesToGraphStrategy:  # Added type annotation
        return SingleNodeHashGraphStrategy()


def get_ts2g_strategy_builder(method: str) -> StrategyBuilder:
    return get_strategy("ts2g_builder", method)