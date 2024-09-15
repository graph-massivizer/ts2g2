import functools

def compare(x, y):
    return x.get_strategy_precedence() - y.get_strategy_precedence()

class StrategyLinkingGraph:
    """
    Links nodes within graph.
    
    **Attributes:**

    - `graph`: networkx.Graph object
    - `strategy_precedence`: tells in which order should the strategies be excetuted

    """
    
    def __init__(self, graph, strategy_precedence):
        self.graph = graph
        self.strategy_precedence = strategy_precedence

    def set_graph(self, graph):
        self.graph = graph

    def get_strategy_precedence(self):
        return self.strategy_precedence

    def apply(self):
        pass

class StrategyLinkingGraphBySeasonalities(StrategyLinkingGraph):
    """
    Links all nodes that are sequentially self.period apart.
    
    **Attributes:**

    - `period`: tells how far apart must two nodes be to be linked

    """
    def __init__(self, period):
        super().__init__(None, 0)
        self.period = period

    def apply(self):
        for i in range(len(self.graph.nodes) - self.period):
            self.graph.add_edge(list(self.graph.nodes)[i], list(self.graph.nodes)[i+self.period], intergraph_binding='seasonality')
        return self.graph


class StrategyLinkingGraphByValue(StrategyLinkingGraph):
    """Links nodes based on their value."""
    def __init__(self, graph):
        super().__init__(graph, 1)
        self.attribute = 'value'

    def apply(self):
        pass


class StrategyLinkingGraphByValueWithinRange(StrategyLinkingGraphByValue):
    """
    Links nodes whose value difference is within range of allowed difference.
    
    **Attributes:**

    - `allowed_difference`: tells us the allowed difference between values of two nodes for the nodes to be linked together
    
    """
    
    def __init__(self, allowed_difference):
        super().__init__(None)
        self.allowed_difference = allowed_difference

    def apply(self):
        for node_11, node_12 in zip(self.graph.nodes(data=True), self.graph.nodes):
            for node_21, node_22 in zip(self.graph.nodes(data=True), self.graph.nodes):
                if  abs(node_11[1][self.attribute][0] - node_21[1][self.attribute][0]) < self.allowed_difference and node_12 != node_22:
                    self.graph.add_edge(node_12, node_22, intergraph_binding = 'timesteps')

        return self.graph

# TODO: this is a builder
class LinkNodesWithinGraph:
    """Builder class for linking nodes within one graph, through which we can access linking strategies."""
    def __init__(self):
        self.graph = None
        self.attribute = 'value'
        self.command_array = []

    def link(self, graph):
        self.graph = graph._get_graph()

        self.command_array.sort(key=functools.cmp_to_key(compare))

        for strat in self.command_array:
            strat.set_graph(self.graph)
            self.graph = strat.apply()

        return self.graph

    def seasonalities(self, period):
        """Notes that we want to connect based on seasonalities, ad sets the period parameter."""
        self.command_array.append(StrategyLinkingGraphBySeasonalities(period))
        return self

    def by_value(self, strategy):
        """Notes that we want to connect nodes based on values and strategy."""
        self.command_array.append(strategy)
        return self