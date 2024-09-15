from deprecated import deprecated
from from_graph.strategy_to_time_sequence import StrategyNextValueInNode
import to_graph.strategy_linking_graph as gs
from from_graph.strategy_to_time_sequence import StrategySelectNextNode

import matplotlib.pyplot as plt
import networkx as nx
import hashlib

import to_graph.strategy_linking_multi_graphs as mgl
import to_graph.strategy_to_graph
from to_graph.strategy_to_graph import BuildStrategyForTimeseriesToGraph
import copy




class Timeseries:
    """Saves extracted data as timeseries."""
    def __init__(self, timeseries):
        self.timeseries = timeseries

    def get_ts(self):
        return self.timeseries
    
    def with_preprocessing(self, strategy):
        return strategy.process(self.timeseries)

class TimeseriesPreprocessing:
    """Processes timeseries."""
    def __init__(self):
        self.ts = None

    def process(self, ts):
        self.ts = ts
        return TimeseriesView([ts])


class TimeseriesPreprocessingSegmentation(TimeseriesPreprocessing):
    """
    Returns a designated segment from timeseries.
    
    **Attributes:**

    - `segmen_start`: start of the segment
    - `segment_end`: end of the segment
    
    """
    def __init__(self, segment_start, segment_end):
        self.seg_st = segment_start
        self.seg_end = segment_end
        self.ts = None

    def process(self, ts):
        self.ts = ts[self.seg_st:self.seg_end]
        return TimeseriesView([self.ts])


class TimeseriesPreprocessingSlidingWindow(TimeseriesPreprocessing):
    """
    Returns an array of segments made with sliding window mechanism.
    
    **Attributes:**

    - `win_size`: size of the creted segments
    - `move_len`: tells for how many data does window move until next segment
    
    """
    def __init__(self, win_size, move_len = 1):
        self.win_size = win_size
        self.move_len = move_len
        self.segments = []

    def process(self, ts):
        if isinstance(ts, list):
            ts = ts[0]
        for i in range(0, len(ts) - self.win_size, self.move_len):
            self.segments.append(ts[i:i + self.win_size])
        return TimeseriesView(self.segments)


class TimeseriesPreprocessingComposite():
    """
    Composites processing strategies, allowing us to use multiple of them.
    
    **Attributes:**

    - `ts`: Timeseries object with extracted timeseries
    """
    def __init__(self):
        self.ts = None
        self.segments = None
        self.strategy = []

    def add(self, strat):
        self.strategy.append(strat)
        return self
    
    def process(self, ts):
        self.ts = ts
        for strat in self.strategy:
            self.ts = strat.process(self.ts).get_ts()[0]
        return TimeseriesView(self.ts)

class TimeseriesStream:
  def read(self):
      return None

class TimeseriesArrayStream(TimeseriesStream):
    def __init__(self, array):
        self.array = copy.deepcopy(array)

    def read(self):
        return copy.deepcopy(self.array)


class TimeseriesView:
    """
    Stores one or more already processed timeseries, then changes them to graph using provided strategy.
    If we have multiple timie series turned to graphs, we can link them into one multivariate graph.
    
    **Attributes:**

    - `ts`: processed timeseries
    - `graph`: networkx.Graph object
    
    """
    
    def __init__(self, ts, attribute = 'value'):
        self.ts = [ts]
        self.graphs = []
        self.attribute = attribute
        self.graph = None
        self.graph_order = []

    def get_ts(self):
        return self.ts

    def add(self, ts):
        series = ts.get_ts()
        for time_ser in series:
            self.ts.append(time_ser)
        return self

    def to_graph(self, strategy: to_graph.strategy_to_graph.BuildStrategyForTimeseriesToGraph):
        for ts in self.ts:
            graph_dict = {}
            order=[]

            counter = 0
            for timeseries in ts:
                g =  strategy.to_graph(TimeseriesArrayStream(timeseries))
                g = g.graph

                for i in range(len(g.nodes)):
                    old_value = g.nodes[i][self.attribute]
                    new_value = [old_value]
                    g.nodes[i][self.attribute] = new_value

                hash = self._hash(g)
                mapping = {node: f"{hash}_{node}" for node in g.nodes}
                g = nx.relabel_nodes(g, mapping)

                nx.set_edge_attributes(g, strategy.get_name(), "strategy")
                graph_dict[self._hash(g) + f"_{counter}"] = g
                order.append(self._hash(g) + f"_{counter}")

                if self.graph == None:
                    self.graph = g

                counter+=1

            self.graphs.append(graph_dict)
            self.graph_order.append(order)


        if len(self.ts) == 1 and len(self.ts[0]) == 1:
            return TimeGraph(self.graph, graphs = self.graphs[0])
        else:
            return self

    def link(self, link_strategy: mgl.LinkGraphs):
        return TimeGraph(link_strategy.link(self.graphs, self.graph_order), graphs = self.graphs)

    def _get_graphs(self):
        return self.graphs

    def _get_graph(self):
        if self.graph == None:
            return TimeGraph(list(self.graphs[0].values())[0])
        else:
            return TimeGraph(self.graph)

    def _hash(self, graph):
        """Returns unique hash of this graph."""
        str_to_hash = str(graph.nodes()) + str(graph.edges())
        return hashlib.md5(str_to_hash.encode()).hexdigest()


# TODO: to be renamed into TimeGraph?
# TODO: we need to delete the TimeGraph object (not this one - the redundant one)?
class TimeGraph:
    """
    Stores already made graph, allows us to add edges and links between nodes.
    
    **Attributes:**

    - `graph`: object networkx.Graph
    
    """
    def __init__(self, graph, graphs = None):
        self.graph = graph
        self.orig_graph = None
        self.graphs = graphs
        self.attribute = 'value'
        self.sequences = None
        self.sequence_visitor = None

    def _get_graph(self):
        return self.graph

    def _get_graphs(self):
        return self.graphs

    def add_edge(self, node_1, node_2, weight=None):
        """Adds edge between node_1 and node_2."""
        if weight == None:
            self.graph.add_edge(list(self.graph.nodes)[node_1], list(self.graph.nodes)[node_2])
        else:
            self.graph.add_edge(list(self.graph.nodes)[node_1], list(self.graph.nodes)[node_2], weight = weight)
        return self

    def link(self, link_strategy: gs.LinkNodesWithinGraph):
        """Links nodes based on link_strategy. link_strategy is object of class Link."""
        self.graph = link_strategy.link(self)
        return self

    def _hash(self):
        """Returns unique hash of this graph."""
        str_to_hash = str(self.graph.nodes()) + str(self.graph.edges())
        return hashlib.md5(str_to_hash.encode()).hexdigest()

    def combine_identical_subgraphs(self):
        """Combines nodes that have same value of attribute self.attribute if graph is classical graph and
        nodes that are identical graphs if graph is created using sliding window mechanism."""
        self.orig_graph = self.graph.copy()
        for j in range(len(self.graphs)):
            for i, node_1 in enumerate(list(self.graphs[j].values())):
                if node_1 not in self.graph:
                    continue

                for node_2 in list(self.graphs[j].values())[i+1:]:
                    if node_2 == None:
                        break
                    if node_2 not in self.graph:
                        continue

                    if(set(list(node_1.edges)) == set(list(node_2.edges))):
                        self.graph = self._combine_subgraphs(self.graph, node_1, node_2, self.attribute)

        return self

    def _combine_subgraphs(self, graph, node_1, node_2, att):
        """Combines nodes node_1 and node_2, that are graphs."""
        for i in range(len(list(node_1.nodes(data=True)))):
            for j in range(len(list(node_2.nodes(data=True))[i][1][att])):
                list(node_1.nodes(data=True))[i][1][att].append(list(node_2.nodes(data=True))[i][1][att][j])

        for neighbor in list(graph.neighbors(node_2)):
            graph.add_edge(node_1, neighbor)

        graph.remove_node(node_2)
        return graph

    def combine_identical_nodes(self):
        """Combines nodes that have same value of attribute self.attribute if graph is classical graph and
        nodes that are identical graphs if graph is created using sliding window mechanism."""
        self.orig_graph = self.graph.copy()

        for i, node_1 in enumerate(list(self.graph.nodes(data=True))):
            if node_1 not in self.graph:
                continue

            for node_2 in list(self.graph.nodes(data=True))[i+1:]:
                if node_2 == None:
                    break
                if node_2 not in self.graph:
                    continue

                if(node_1[self.attribute] == node_2[self.attribute]):
                    self.graph = self._combine_nodes(self.graph, node_1, node_2, self.attribute)

        return self

    def _combine_nodes(self, graph, node_1, node_2, att):
        """Combines nodes node_1 and node_2."""
        node_1[att].append(node_2[att])
        for neighbor in list(graph.neighbors(node_2)):
            graph.add_edge(node_1, neighbor)

        graph.remove_node(node_2)
        return graph

    def draw(self, color = "black"):
        """Draws the created graph"""
        pos=nx.spring_layout(self.graph, seed=1)
        nx.draw(self.graph, pos, node_size=40, node_color=color)
        plt.show()
        return self
    
    def to_sequence(self, sequence_visitor):
        self.sequence_visitor = sequence_visitor
        self.sequences = sequence_visitor.to_sequence(self)
        return self
    
    def draw_sequence(self):
        """Draws timeseries."""

        colors = []
        for j in range(len(self.sequences)):
            colors.append("black")

        for j in range(len(self.sequences)):
            self.plot_timeseries(self.sequences[j], f"next_node_strategy = {self.sequence_visitor._get_node_strategy_name()}, next_value = {self.sequence_visitor._get_value_strategy_name()}", "Date", "Value", colors[j])
        plt.show()
        return self

    def plot_timeseries(self, sequence, title, x_legend, y_legend, color):
        """Function to sets parameters to draw timeseries."""
        plt.figure(figsize=(10, 6))
        plt.plot(sequence, linestyle='-', color=color)

        plt.title(title)
        plt.xlabel(x_legend)
        plt.ylabel(y_legend)
        plt.grid(True)


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
        pass

    def _is_equal(self, graph_1, graph_2):
        """Compares two graphs if they are equal."""
        if(graph_1.nodes != graph_2.nodes): return False
        if(graph_1.edges != graph_2.edges): return False
        for i in range(len(graph_1.nodes)):
            if list(list(graph_1.nodes(data=True))[i][1][self.att]) != list(list(graph_2.nodes(data=True))[i][1][self.att]):
                    return False
        return True


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
                for i in range(len(self.nodes[j])):
                    if(self._is_equal(current_nodes[j], list(self.graph.nodes)[i])):
                        index = i
                        break

                self.sequences[j] = self.value_strategy.append(self.sequences[j], current_nodes[j], j, index)
                if self.sequences[j][-1] == None:
                    return self

            for j in range(self.skip_values + 1):
                for k in range(len(current_nodes)):

                    current_nodes[k] = self.node_strategy.next_node(ts_len, k, current_nodes, self.switch_graphs, current_nodes[0])

                    if(current_nodes[k] == None):
                        return self

            ts_len += 1
        return self.sequences

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

                for i in range(len(list(self.nodes[j]))):
                    if(current_nodes_data[j] == self.data_nodes[j][i]):
                        index = i
                        break

                self.sequences[j] = self.value_strategy.append(self.sequences[j], current_nodes_data[j], j, index)
                if self.sequences[j][-1] == None:
                    return

            for j in range(self.skip_values+1):
                for k in range(len(current_nodes)):
                    current_nodes[k] = self.node_strategy.next_node(i, k, current_nodes, self.switch_graphs, current_nodes[0])

                    new_index = self.nodes[k].index(current_nodes[k])
                    current_nodes_data[k] = self.data_nodes[k][new_index]
                    if(current_nodes[k] == None):
                        break

            i += 1
        return self.sequences



