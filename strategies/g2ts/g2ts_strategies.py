from .strategy_to_time_sequence import StrategySelectNextNode, StrategyNextValueInNode
from strategies.register import register_strategy, get_strategy

class ToSequenceVisitorMaster:
    """
    Turns graphs back to timeseries.
    
    **Attributes:**

    - `graph`: networkx.Graph object
    - `node_strategy`: strategy_to_time_sequence.StrategySelectNextNode object
    - `value_strategy`: strategy_to_time_sequence.StrategyNextValueInNode object

    """
    def __init__(self):
        self.ts_view = None
        self.graph = None
        self.node_strategy = None
        self.value_strategy = None
        self.skip_values = 0
        self.timeseries_len = 100
        self.sequences = None
        self.switch_graphs = 1
        self.nodes = None
        self.data_nodes = None
        self.att = 'value'

    def _set_nodes(self, dict: dict):
        pass

    def set_attribute(self, att):
        self.att = att

    """Next 4 function set the parameters, that are later on used as a strategy for converting graph to timeseries."""
    """--->"""
    def next_node_strategy(self, strategy: StrategySelectNextNode):
        self.node_strategy = strategy
        self.switch_graphs = strategy.get_change()
        return self

    def next_value_strategy(self, strategy: StrategyNextValueInNode):
        self.value_strategy = strategy
        self.skip_values = strategy.get_skip()
        return self

    def ts_length(self, x):
        self.timeseries_len = x
        return self
    """<---"""

    def _get_node_strategy_name(self):
        return self.node_strategy.get_name()
    
    def _get_value_strategy_name(self):
        return self.value_strategy.get_name()

    def to_sequence(self, graph):
        """Converts graph into time sequences."""
        raise NotImplementedError("Subclasses must implement to_sequence(graph)")

    def _is_equal(self, graph_1, graph_2):
        """Compares two graphs if they are equal."""
        if(graph_1.nodes != graph_2.nodes): return False
        if(graph_1.edges != graph_2.edges): return False
        for i in range(len(graph_1.nodes)):
            if list(list(graph_1.nodes(data=True))[i][1][self.att]) != list(list(graph_2.nodes(data=True))[i][1][self.att]):
                    return False
        return True


@register_strategy("g2ts", "sliding_window")
class ToSequenceVisitorSlidingWindow(ToSequenceVisitorMaster):
    """Converts graphs made using sliding window mechanism back to timeseries"""
    def __init__(self):
        super().__init__()

    def _set_nodes(self, dicts: dict):

        if isinstance(dicts, list):
            graphs = [{} for _ in range(len(dicts))]
            for i in range(len(dicts)):
                for j in range(len(dicts[i].values())):
                    graphs[i][list(dicts[i].items())[j]] = list(dicts[i].values())[j]
            dicts = graphs

        self.nodes = [[] for _ in range(len(dicts))]

        for i in range(len(dicts)):
            for graph in dicts[i].values():
                self.nodes[i].append(graph)
        return self

    def to_sequence(self, graph):
        
        self.ts_view = graph
        self.graph = graph._get_graph()
        self._set_nodes(graph._get_graphs())


        self.sequences = [[] for _ in range(len(self.nodes))]

        current_nodes = [None for _ in range(len(self.nodes))]

        for i in range(len(self.nodes)):
            current_nodes[i] = list(self.nodes[i])[0]

        dictionaries = [{} for _ in range(len(self.nodes))]
        for i in range(len(self.nodes)):
            for j in range(len(list(self.nodes[i]))):
                dictionaries[i][j] = 0


        self.value_strategy.set_arguments(dictionaries, self.att)
        self.node_strategy.set_arguments(self.graph, self.nodes, dictionaries, self.att)


        ts_len = 0
        while len(self.sequences[0]) < self.timeseries_len:
            for j in range(len(self.sequences)):
                
                index = 0
                for i in range(len(list(self.nodes[j]))):
                
                    if(self._is_equal(current_nodes[j], list(self.graph.nodes)[i])):
                        index = i
                        break

                self.sequences[j] = self.value_strategy.append(self.sequences[j], current_nodes[j], j, index)
                if self.sequences[j][-1] == None:
                    return self

            for j in range(self.skip_values + 1):
                for k in range(len(current_nodes)):

                    current_nodes[k] = self.node_strategy.next_node(ts_len, k, current_nodes, self.switch_graphs, current_nodes[k])

                    if(current_nodes[k] == None):
                        return self

            ts_len += 1
        return self.sequences

@register_strategy("g2ts", "plain")
class ToSequenceVisitor(ToSequenceVisitorMaster):
    """Converts ordinary graphs back to timeseries."""
    def __init__(self):
        super().__init__()

    def _set_nodes(self, dict: dict):

        if isinstance(dict, list):
            graphs = {}
            for i in range(len(dict)):
                graphs[list(dict[i].items())[0]] = list(dict[i].values())[0]
            dict = graphs

        self.nodes = []
        self.data_nodes = []
        for graph in dict.values():
            self.nodes.append(list(graph.nodes))
            self.data_nodes.append(list(graph.nodes(data=True)))
        return self

    def to_sequence(self, graph):

        self.ts_view = graph
        self.graph = graph._get_graph()
        self._set_nodes(graph._get_graphs())

        self.sequences = [[] for _ in range(len(self.nodes))]

        current_nodes = [None for _ in range(len(self.nodes))]
        current_nodes_data = [None for _ in range(len(self.data_nodes))]

        for i in range(len(self.nodes)):
            current_nodes[i] = self.nodes[i][0]
            current_nodes_data[i] = self.data_nodes[i][0]

        dictionaries = [{} for _ in range(len(self.nodes))]
        for i in range(len(self.nodes)):
            for j in range(len(list(self.nodes[i]))):
                dictionaries[i][j] = 0

        self.value_strategy.set_arguments(dictionaries, self.att)
        self.node_strategy.set_arguments(self.graph, self.nodes, dictionaries, self.att)

        i = 0
        while len(self.sequences[0]) < self.timeseries_len:

            for j in range(len(current_nodes)):

                index = 0

                for b in range(len(list(self.nodes[j]))):
                    if(current_nodes_data[j] == self.data_nodes[j][b]):
                        index = b
                        break

                self.sequences[j] = self.value_strategy.append(self.sequences[j], current_nodes_data[j], j, index)
                if self.sequences[j][-1] == None:
                    return
                
            for j in range(self.skip_values+1):
                for k in range(len(current_nodes)):
                    current_nodes[k] = self.node_strategy.next_node(i, k, current_nodes, self.switch_graphs, current_nodes[k])
                    
                    new_index = self.nodes[k].index(current_nodes[k])
                    current_nodes_data[k] = self.data_nodes[k][new_index]
                    if(current_nodes[k] == None):
                        break

            i += 1
        return self.sequences



@register_strategy("g2ts", "ordinal_partition")
class ToSequenceVisitorOrdinalPartition(ToSequenceVisitorMaster):
    """Converts graphs made using ordinal partition mechanism back to timeseries"""
    def __init__(self):
        super().__init__()
        self.histogram_frequencies = None
        self.histogram_bins = None
        self.w = 1
        self.tau = 1

    def _set_nodes(self, dict: dict):

        if isinstance(dict, list):
            graphs = {}
            for i in range(len(dict)):
                graphs[list(dict[i].items())[0]] = list(dict[i].values())[0]
            dict = graphs

        self.nodes = []
        self.data_nodes = []
        for graph in dict.values():
            self.nodes.append(list(graph.nodes))
            self.data_nodes.append(list(graph.nodes(data=True)))
        return self
    
    def to_sequence(self, graph):


        self.graph = graph._get_graph()

        self.histogram_frequencies, self.histogram_bins = graph._get_histogram()
        self._set_nodes(graph._get_graphs())
        self.w, self.tau = graph._get_w_tau()
        one_ts_length = self.timeseries_len/self.tau
        short_series = [[[] for i in range(self.tau)] for i in range(len(graph._get_graphs()))]
        current_nodes = [None for _ in range(len(self.nodes))]
        current_nodes_data = [None for _ in range(len(self.data_nodes))]
        for i in range(len(self.nodes)):
            current_nodes[i] = self.nodes[i][0]
            current_nodes_data[i] = self.data_nodes[i][self.nodes[i].index(current_nodes[i])]

        self.node_strategy.set_arguments(self.graph, self.nodes, {}, self.att)

        i = 0
        while(len(short_series[0][0]) < one_ts_length):
            for k in range(len(graph._get_graphs())):
                for i in range(self.tau):
                    if(len(short_series[k][i]) == 0):
                        short_series[k][i] = self.value_strategy.append_start(short_series[k][i], k, current_nodes_data[k], self.histogram_frequencies, self.histogram_bins, self.w)
                    else:
                        short_series[k][i] = self.value_strategy.append(short_series[k][i], k, current_nodes_data[k], self.histogram_frequencies, self.histogram_bins, self.w)
                    if(i < self.tau-1):
                        current_nodes[k] = self.node_strategy.next_node(i, k, current_nodes, self.switch_graphs, current_nodes[k])
                        current_nodes_data[k] = self.data_nodes[k][self.nodes[k].index(current_nodes[k])]
            
            for j in range(self.skip_values+1):
                for k in range(len(graph._get_graphs())):
                    current_nodes[k] = self.node_strategy.next_node(i, k, current_nodes, self.switch_graphs, current_nodes[k])
                    current_nodes_data[k] = self.data_nodes[k][self.nodes[k].index(current_nodes[k])]
            
            i+=1
        
        
        self.sequences = [[] for i in range(len(graph._get_graphs()))]
        for k in range(len(graph._get_graphs())):
            for j in range(self.tau):
                for i in range(len(short_series[k][j])):
                    self.sequences[k].append(short_series[k][j][i])
        
        return self.sequences

@register_strategy("g2ts", "quantile")
class ToSequenceVisitorQuantile(ToSequenceVisitorMaster):
    def __init__(self):
        super().__init__()
        self.bins = None

    def _set_nodes(self, dict: dict):

        if isinstance(dict, list):
            graphs = {}
            for i in range(len(dict)):
                graphs[list(dict[i].items())[0]] = list(dict[i].values())[0]
            dict = graphs

        self.nodes = []
        self.data_nodes = []
        for graph in dict.values():
            self.nodes.append(list(graph.nodes))
            self.data_nodes.append(list(graph.nodes(data=True)))
        return self
    
    def to_sequence(self, graph):
        self.graph = graph._get_graph()
        self.bins, self.values = graph._get_quantiles()
        self._set_nodes(graph._get_graphs())

        self.sequences = [[] for _ in range(len(self.nodes))]

        current_nodes = [None for _ in range(len(self.nodes))]
        current_nodes_data = [None for _ in range(len(self.data_nodes))]

        self.node_strategy.set_arguments(self.graph, self.nodes, {}, self.att)

        for i in range(len(self.nodes)):
            current_nodes[i] = self.nodes[i][0]
            current_nodes_data[i] = self.data_nodes[i][0]

        
        i = 0
        while len(self.sequences[0]) < self.timeseries_len:

            for j in range(len(current_nodes)):

                self.sequences[j] = self.value_strategy.append(self.sequences[j], current_nodes_data[j], self.bins[j], self.values[j])
                
                
            for j in range(self.skip_values+1):
                for k in range(len(current_nodes)):
                    current_nodes[k] = self.node_strategy.next_node(i, k, current_nodes, self.switch_graphs, current_nodes[k])
                    
                    new_index = self.nodes[k].index(current_nodes[k])
                    current_nodes_data[k] = self.data_nodes[k][new_index]
                    if(current_nodes[k] == None):
                        break

            i += 1
        return self.sequences

# ---- Factory for G->TS strategies ----
def get_g2ts_strategy(method: str):
    """Fetch a registered graph-to-timeseries strategy by name.
    Valid names (by default): 'plain', 'sliding_window', 'ordinal_partition', 'quantile'.
    """
    return get_strategy("g2ts", method)