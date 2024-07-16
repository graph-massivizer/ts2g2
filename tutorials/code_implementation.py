import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

#import os
import csv
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from timeseries.strategies import TimeseriesToGraphStrategy, TimeseriesEdgeVisibilityConstraintsNatural, TimeseriesEdgeVisibilityConstraintsHorizontal, EdgeWeightingStrategyNull, TimeseriesEdgeVisibilityConstraintsVisibilityAngle
from generation.strategies import RandomWalkWithRestartSequenceGenerationStrategy, RandomWalkSequenceGenerationStrategy, RandomNodeSequenceGenerationStrategy, RandomNodeNeighbourSequenceGenerationStrategy, RandomDegreeNodeSequenceGenerationStrategy
from core import model 
from sklearn.model_selection import train_test_split
import itertools

import xml.etree.ElementTree as ET
import random
import hashlib

class TSprocess:

    def __init__(self):
        pass

    def process(time_series):
        pass

class Segment(TSprocess):

    def __init__(self, segment_start, segment_end):
        self.segment_start = segment_start
        self.segment_end = segment_end
    
    def process(self, time_series):
        return TimeSeries(time_series[self.segment_start:self.segment_end])

class SlidingWindow(TSprocess):

    def __init__(self, window_size, win_move_len = 1):
        self.window_size = window_size
        self.win_move_len = win_move_len
    
    def process(self, time_series):
        segments = []
        for i in range(0, len(time_series) - self.window_size, self.win_move_len):
            segments.append(time_series[i:i + self.window_size])
        
        new_series = []
        for i in range(len(segments)):
            new_series.append(TimeSeries(segments[i]))
        
        return TimeSeries(new_series, True)




class Link:
    def __init__(self, graph = None, multi = False):
        self.seasonalites = False
        self.same_timestep = -1
        self.graph = graph
        self.coocurrence = False
        self.multi = multi
        self.period = None
    
    def seasonalities(self, period):
        self.seasonalites = True
        self.period = period
        return self
    
    def same_timesteps(self, allowed_difference):
        self.same_timestep = allowed_difference
        return self

    def time_coocurence(self):
        self.coocurrence = True
        return self
    
    def link_positional(self, stud):
        g = None
        if self.multi:
            g = nx.MultiGraph()
        else :
            g = nx.Graph()

        min_size = None
        
        for graph in stud.values():
            if min_size == None or len(graph.nodes) < min_size:
                min_size = len(graph.nodes)

        for hash, graph in stud.items():
            nx.set_node_attributes(graph, hash, 'id')
            i = 0
            for node in list(graph.nodes(data = True)):
                node[1]['order'] = i
                i += 1
        
        for graph in stud.values():
            g = nx.compose(g, graph)

        i = 0
        j = 0
        for (node_11, node_12) in zip(list(g.nodes(data = True)), list(g.nodes)):
            
            i = 0
            for (node_21, node_22) in zip(list(g.nodes(data = True)), list(g.nodes)):
                if i == j:
                    i+=1
                    continue

                if node_11[1]['order'] == node_21[1]['order']:
                    g.add_edge(node_12, node_22, intergraph_binding = 'positional')
                i+=1
            j+=1
        
        self.graph = g

    def link_seasonalities(self):
        for i in range(list(self.graph.nodes) - self.period):
            self.graph.add_edge(list(self.graph.nodes)[i], list(self.graph.nodes)[i+self.period], intergraph_binding='seasonality')

    def link_same_timesteps(self):
        for node_11, node_12 in zip(self.graph.nodes(data=True), self.graph.nodes):
            for node_21, node_22 in zip(self.graph.nodes(data=True), self.graph.nodes):
                if  abs(node_11[1]['value'][0] - node_21[1]['value'][0]) < self.same_timestep and node_12 != node_22:
                    self.graph.add_edge(node_12, node_22, intergraph_binding = 'timesteps')

    def link(self, stud):
        self.graph = stud

        if self.seasonalites:
            self.link_seasonalities(self.graph)

        if self.same_timestep > 0:
            self.link_same_timesteps()
        
        if self.coocurrence:
            self.link_positional(stud)

        return self.graph




class Strategy:

    def __init__(self):
        self.visibility = []
        self.graph_type = "undirected"
        self.edge_weighting_strategy = EdgeWeightingStrategyNull()
        self.str_name = None

    def with_angle(self, angle):
        self.visibility.append(TimeseriesEdgeVisibilityConstraintsVisibilityAngle(angle))
        self.str_name += (f" with angle({angle})")
        return self

    def with_limit(self, limit):
        pass
    
    def strategy_name(self):
        return self.str_name

    def get_strategy(self):
        return TimeseriesToGraphStrategy(
            visibility_constraints = self.visibility,
            graph_type= self.graph_type,
            edge_weighting_strategy=self.edge_weighting_strategy
        )

class NaturalVisibility(Strategy):

    def __init__(self):
        super().__init__()
        self.visibility = [TimeseriesEdgeVisibilityConstraintsNatural()]
        self.str_name = "Natural visibility strategy"
    
    def with_limit(self, limit):
        self.visibility[0] = TimeseriesEdgeVisibilityConstraintsNatural(limit)
        self.str_name += (f" with limit({limit})")
        return self
    
class HorizontalVisibility(Strategy):

    def __init__(self):
        super().__init__()
        self.visibility = [TimeseriesEdgeVisibilityConstraintsHorizontal()]
        self.str_name = "Horizontal visibility strategy"
    
    def with_limit(self, limit):
        self.visibility[0] = TimeseriesEdgeVisibilityConstraintsHorizontal(limit)
        self.str_name += (f" with limit({limit})")
        return self




class CsvRead:
    def __init__(self):
        pass

    def from_csv(self):
        pass

class CsvStock(CsvRead):
    def __init__(self, path, y_column):
        self.path = path
        self.y_column = y_column
    
    def from_csv(self):
        time_series = pd.read_csv(self.path)
        time_series["Date"] = pd.to_datetime(time_series["Date"])
        time_series.set_index("Date", inplace=True)
        time_series = time_series[self.y_column]
        return time_series

class XmlRead:
    def __init__(self):
        pass

    def from_xml(self):
        pass

class XmlSomething(XmlRead):
    def __init__(self, path, item, season = "Annual"):
        self.path = path
        self.item = item
        self.season = season
    
    def from_xml(self):
        tree = ET.parse(self.path)
        root = tree.getroot()

        financial_statements = root.find('FinancialStatements')
        COAMap = financial_statements.find('COAMap')
        
        periods = None
        if self.season.lower() == "annual":
            periods = financial_statements.find("AnnualPeriods")
        else:
            periods = financial_statements.find('InterimPeriods')

        elements = periods.findall(f".//lineItem[@coaCode = '{self.item}']")
        column = []

        for element in elements:
            column.append(float(element.text))
        
        return column





class TimeSeries():
    
    def __init__(self, time_series = None, slid_win = False, attribute = "value"):
        self.time_series = time_series
        self.strategy = None
        self.graph = None
        self.slid_win = slid_win
        self.slid_graphs = []
        self.attribute = attribute
    
    def from_csv(self, csv_read):
        self.time_series = csv_read.from_csv()
        return self
    
    def from_xml(self, xml_read):
        self.time_series = xml_read.from_xml()
        return self

    def return_graph(self):
        return self.graph

    def process(self, ts_processing_strategy = None):
        if ts_processing_strategy == None:
            return self

        return ts_processing_strategy.process(self.time_series)
        #to do: how to return efficiently

    def to_graph(self, strategy):
        self.strategy = strategy.get_strategy()

        if self.slid_win:
            self.graph = nx.MultiGraph()
            for i in range(len(self.time_series)):
                self.slid_graphs.append(self.time_series[i].to_graph(strategy).return_graph())
            
            for i in range(len(self.slid_graphs)-1):
                self.graph.add_edge(self.slid_graphs[i], self.slid_graphs[i+1])
            
            for graph in self.graph.nodes:
                for i in range(len(graph.nodes)):
                    old_value = list(graph.nodes(data = True))[i][1][self.attribute]
                    new_value = [old_value]
                    list(graph.nodes(data=True))[i][1][self.attribute] = new_value
        
                
        else:
            g =  self.strategy.to_graph(model.TimeseriesArrayStream(self.time_series))
            self.graph = g.graph

            for i in range(len(self.graph.nodes)):
                old_value = self.graph.nodes[i][self.attribute]
                new_value = [old_value]
                self.graph.nodes[i][self.attribute] = new_value
            

            hash = self.hash()
            mapping = {node: f"{hash}_{node}" for node in self.graph.nodes}
            self.graph = nx.relabel_nodes(self.graph, mapping)

            nx.set_edge_attributes(self.graph, strategy.strategy_name(), "strategy")


        return self
    
    def combine_identical_nodes(self):

        if self.slid_win:

            for i, node_1 in enumerate(list(self.graph.nodes)):
                if node_1 not in self.graph:
                    continue

                for node_2 in list(self.graph.nodes)[i+1:]:
                    if node_2 == None:
                        break
                    if node_2 not in self.graph:
                        continue

                    if(set(list(node_1.edges)) == set(list(node_2.edges))):
                        self.graph = combine_nodes_win(self.graph, node_1, node_2, self.attribute)
        else:

            for i, node_1 in enumerate(list(self.graph.nodes(data=True))):
                if node_1 not in self.graph:
                    continue

                for node_2 in list(self.graph.nodes(data=True))[i+1:]:
                    if node_2 == None:
                        break
                    if node_2 not in self.graph:
                        continue

                    if(node_1[self.attribute] == node_2[self.attribute]):
                        self.graph = combine_nodes(self.graph, node_1, node_2, self.attribute)
            

        return self
        #to do: else: combines nodes with same attribute
    
    def draw(self, color = "black"):
        pos=nx.spring_layout(self.graph, seed=1)
        nx.draw(self.graph, pos, node_size=40, node_color=color)
        plt.show()
        return self
    
    def link(self, link_strategy):
        self.graph = link_strategy.link(self.graph)
        return self
        
    def add_edge(self, node_1, node_2, weight=None):
        if weight == None:
            self.graph.add_edge(list(self.graph.nodes)[node_1], list(self.graph.nodes)[node_2])
        else:
            self.graph.add_edge(list(self.graph.nodes)[node_1], list(self.graph.nodes)[node_2], weight = weight)
        return self
    
    def hash(self):
        str_to_hash = str(self.graph.nodes()) + str(self.graph.edges())
        return hashlib.md5(str_to_hash.encode()).hexdigest()




class GraphMaster:
    def __init__(self, graph, strategy):
        self.graph = graph
        self.next_node_strategy = "random"
        self.next_value_strategy = "random"
        self.skip_values = 0
        self.time_series_len = 100
        self.sequences = None
        self.walk = "one"
        self.switch_graphs = 1
        self.colors = None
        self.nodes = None
        self.data_nodes = None
        self.strategy = strategy
    
    def set_nodes(self, nodes, data_nodes):
        pass
    
    def walk_through_all(self):
        self.walk = "all"
        return self
    
    def change_graphs_every_x_steps(self, x):
        self.switch_graphs = x
        return self
    
    def choose_next_node(self, strategy):
        self.next_node_strategy = strategy
        return self
    
    def choose_next_value(self, strategy):
        self.next_value_strategy = strategy
        return self
    
    def skip_every_x_steps(self, x):
        self.skip_values = x
        return self
    
    def ts_length(self, x):
        self.time_series_len = x
        return self
    
    def to_time_sequence(self):
        pass
    
    def to_multiple_time_sequences(self):
        pass

    def is_equal(self, graph_1, graph_2):
        if(graph_1.nodes != graph_2.nodes): return False
        if(graph_1.edges != graph_2.edges): return False
        for i in range(len(graph_1.nodes)):
            if list(list(graph_1.nodes(data=True))[i][1]['value']) != list(list(graph_2.nodes(data=True))[i][1]['value']):
                    return False
        return True

    def plot_timeseries(self, sequence, title, x_legend, y_legend, color):
        plt.figure(figsize=(10, 6))
        plt.plot(sequence, linestyle='-', color=color)
        
        plt.title(title)
        plt.xlabel(x_legend)
        plt.ylabel(y_legend)
        plt.grid(True)

    def draw(self):
        if self.colors == None:
            self.colors = []
            for j in range(len(self.sequences)):
                self.colors.append("black")
        
        for j in range(len(self.sequences)):
            self.plot_timeseries(self.sequences[j], f"walk = {self.walk}, next_node_strategy = {self.next_node_strategy}, value = {self.next_value_strategy}", "Date", "Value", self.colors[j])
        plt.show()

class GraphSlidWin(GraphMaster):
    def __init__(self, graph):
        super().__init__(graph, "slid_win")
    
    def set_nodes(self, nodes):
        self.nodes = nodes
        return self

    def to_time_sequence(self):
        self.nodes = [list(self.nodes)]
        return self.to_multiple_time_sequences()

    def to_multiple_time_sequences(self):
    
        self.sequences = [[] for _ in range(len(self.nodes))]

        current_nodes = [None for _ in range(len(self.nodes))]

        for i in range(len(self.nodes)):
            current_nodes[i] = self.nodes[i][0]
        
        dictionaries = [{} for _ in range(len(self.nodes))]
        for i in range(len(self.nodes)):
            for j in range(len(list(self.nodes[i]))):
                dictionaries[i][j] = 0

        
        strategy = None

        strategy = ChooseStrategySlidWin(self.walk, self.next_node_strategy, self.next_value_strategy, self.graph, self.nodes, dictionaries)

        i = 0
        while len(self.sequences[0]) < self.time_series_len:
            
            for j in range(len(self.sequences)):

                index = 0
                for i in range(len(list(self.nodes[j]))):
                    if(self.is_equal(current_nodes[j], list(self.graph.nodes)[i])):
                        index = i
                        break

                self.sequences[j] = strategy.append(self.sequences[j], current_nodes[j], j, index)
                if self.sequences[j][-1] == None:
                    return

            for j in range(self.skip_values):
                for k in range(len(current_nodes)):
                    current_nodes[k] = strategy.next_node(i, k, current_nodes, self.switch_graphs)
                    if(current_nodes[k] == None):
                        break
            
            for k in range(len(current_nodes)):
                    current_nodes[k] = strategy.next_node(i, k, current_nodes, self.switch_graphs)
                    if(current_nodes[k] == None):
                        break
            
            i += 1
        return self

class Graph(GraphMaster):
    def __init__(self, graph):
        super().__init__(graph, "classic")
    
    def set_nodes(self, nodes, data_nodes):
        self.nodes = nodes
        self.data_nodes = data_nodes
        return self

    def to_time_sequence(self):
        self.nodes = [list(self.nodes)]
        self.data_nodes = [list(self.data_nodes)]
        return self.to_multiple_time_sequences()

    def to_multiple_time_sequences(self):
    
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

        strategy = ChooseStrategy(self.walk, self.next_node_strategy, self.next_value_strategy, self.graph, self.nodes, dictionaries)

        i = 0
        while len(self.sequences[0]) < self.time_series_len:
            
            for j in range(len(current_nodes)):

                index = 0

                for i in range(len(list(self.nodes[j]))):
                    if(current_nodes_data[j] == self.data_nodes[j][i]):
                        index = i
                        break
                
                self.sequences[j] = strategy.append(self.sequences[j], current_nodes_data[j], j, index)
                if self.sequences[j][-1] == None:
                    return

            for j in range(self.skip_values):
                for k in range(len(current_nodes)):
                    current_nodes[k] = strategy.next_node(i, k, current_nodes, self.switch_graphs)

                    new_index = self.nodes[k].index(current_nodes[k])
                    current_nodes_data[k] = self.data_nodes[k][new_index]
                    if(current_nodes[k] == None):
                        break
            
            for k in range(len(current_nodes)):
                    current_nodes[k] = strategy.next_node(i, k, current_nodes, self.switch_graphs)
                    if(current_nodes[k] == None):
                        break
            
            i += 1
        return self

class ChooseStrategyMaster:
    def __init__(self, walk, next_node_strategy, value, graph, nodes, dictionaries):
        self.next_node_strategy = next_node_strategy
        self.walk = walk
        self.value = value
        self.graph = graph
        self.nodes = nodes
        self.dictionaries = dictionaries
    
    def append_random(self, sequence, graph):
        pass

    def append_lowInd(self, sequence, graph, graph_index, index):
        pass

    def append(self, sequence, graph, graph_index, index):
        if(self.value) == "random":
            return self.append_random(sequence, graph)
        elif self.value == "sequential" :
           return self.append_lowInd(sequence, graph, graph_index, index)
        else:
            print("you chose non-existent method of value selection")
            print("please choose between: random")
            return None
    
    def next_node_one_random(self, graph_index, node):
        neighbors = set(self.graph.neighbors(node))
        neighbors = list(set(self.nodes[graph_index]) & neighbors)
        return random.choice(neighbors)

    def next_node_one_weighted(self, graph_index, node):
        neighbors = set(self.graph.neighbors(node))
        neighbors = list(set(self.nodes[graph_index]) & neighbors)
        
        weights = []
        total = 0
        for neighbor in neighbors:
            num = self.graph.number_of_edges(node, neighbor)
            weights.append(num)
            total += num
        for element in weights:
            element /= total
        
        return random.choices(neighbors, weights=weights, k=1)[0]

    def next_node_one(self, graph_index, node):
        if self.next_node_strategy == "random":
            return self.next_node_one_random(graph_index, node)
        elif  self.next_node_strategy == "weighted":
            return self.next_node_one_weighted(graph_index, node)
        else:
            print("you chose non-existent next_node_strategy.")
            print("please choose between: random, weighted")
            return None
    
    def next_node_all_random(self, i, graph_index, nodes, switch):
        index = int((i/switch) % len(nodes))
        neighbors = set(self.graph.neighbors(nodes[index]))
        
        neighbors = list(set(self.nodes[graph_index]) & neighbors)
        return random.choice(neighbors)
    
    def next_node_all_weighted(self, i, graph_index, nodes, switch):
        
        index = int((i/switch) % len(nodes))
        neighbors = set(self.graph.neighbors(nodes[index]))
        neighbors = list(set(self.nodes[graph_index]) & neighbors)
        
        weights = []
        total = 0

        for neighbor in neighbors:
            num = self.graph.number_of_edges(nodes[index], neighbor)
            weights.append(num)
            total += num
        
        for element in weights:
            element /= total
        
        return random.choices(neighbors, weights=weights, k=1)[0]
    
    def next_node_all(self, i, graph_index, nodes, switch):
        if self.next_node_strategy == "random":
            return self.next_node_all_random(i, graph_index, nodes, switch)
        elif  self.next_node_strategy == "weighted":
            return self.next_node_all_weighted(i, graph_index, nodes, switch)
        else:
            print("you chose non-existent next_node_strategy.")
            print("please choose between: random, weighted")
            return None
    
    def next_node(self, i, graph_index, nodes, switch):
        if self.walk == "one":
            return self.next_node_one(graph_index, nodes[0])
        elif self.walk == "all":
            return self.next_node_all(i, graph_index, nodes, switch)
        else:
            print("you chose non-existent walk")
            print("please choose between: one, all")
            return None
    
class ChooseStrategy(ChooseStrategyMaster):
    def __init__(self, walk, next_node_strategy, value, graph, nodes, dictionaries):
        super().__init__(walk, next_node_strategy, value, graph, nodes, dictionaries)
    
    def append_random(self, sequence, graph):
        index = random.randint(0, len(graph[1]['value']) - 1)
        sequence.append(graph[1]['value'][index])
        return sequence
    
    def append_lowInd(self, sequence, graph, graph_index, index):
        if int(self.dictionaries[graph_index][index]/2) >= len(list(graph(data=True)[1]['value'])):
            self.dictionaries[graph_index][index] = 0
        
        ind = int(self.dictionaries[graph_index][index]/2)
        sequence.append(graph(data = True)[1]['value'][ind])
        self.dictionaries[graph_index][index] += 1
        return sequence

class ChooseStrategySlidWin(ChooseStrategyMaster):
    def __init__(self, walk, next_node_strategy, value, graph, nodes, dictionaries):
        super().__init__(walk, next_node_strategy, value, graph, nodes, dictionaries)
    
    def append_random(self, sequence, graph):

        nodes = list(graph.nodes(data = True))
        random.shuffle(nodes)

        for node in nodes:
            index = random.randint(0, len(node[1]['value']) - 1)
            sequence.append(node[1]['value'][index])
        return sequence

    def append_lowInd(self, sequence, graph, graph_index, index):
        
        if int(self.dictionaries[graph_index][index]/2) >= len(list(graph.nodes(data=True)[0]['value'])):
            self.dictionaries[graph_index][index] = 0
    
        ind = int(self.dictionaries[graph_index][index]/2)

        for node in graph.nodes(data=True):
            sequence.append(node[1]['value'][ind])
    
        self.dictionaries[graph_index][index] += 1
        return sequence


class MultivariateTimeSeries:
    def __init__(self):
        self.graphs = {}
        self.multi_graph = None
        self.attribute = 'value'
    
    def set_attribute(self, att):
        self.attribute = att
        return self
    
    def link(self, link_strategy):
        self.multi_graph = link_strategy.link(self.graphs)
        return self
    
    def return_graph(self):
        return self.multi_graph

    def add(self, time_serie):
        self.graphs[time_serie.hash()] = time_serie.return_graph()
        return self
    
    def combine_identical_nodes_win(self):
        for graph in self.graphs.values():

            for i, node_1 in enumerate(list(graph.nodes)):
                if node_1 not in graph:
                    continue

                for node_2 in list(graph.nodes)[i+1:]:
                    if node_2 == None:
                        break
                    if node_2 not in graph:
                        continue

                    if(set(list(node_1.edges)) == set(list(node_2.edges))):
                        graph = combine_nodes_win(graph, node_1, node_2, self.attribute)
        return

    def combine_identical_nodes(self):
        if isinstance(self.multi_graph, nx.MultiGraph):
            self.combine_identical_nodes_win()
            return self
        
        for graph in self.graphs.values():

            for i, node_1 in enumerate(list(graph.nodes(data=True))):
                if node_1 not in graph:
                    continue

                for node_2 in list(graph.nodes(data=True))[i+1:]:
                    if node_2 == None:
                        break
                    if node_2 not in graph:
                        continue

                    if(node_1[self.attribute] == node_2[self.attribute]):
                        graph = combine_nodes(graph, node_1, node_2, self.attribute)
        return self
    
    def get_graph_nodes(self):
        nodes = []
        for graph in self.graphs.values():
            nodes.append(list(graph.nodes))
        
        return nodes

    def get_graph_nodes_data(self):
        nodes = []
        for graph in self.graphs.values():
            nodes.append(list(graph.nodes(data = True)))
        
        return nodes
    
    def draw(self, color = "black"):
        pos=nx.spring_layout(self.multi_graph, seed=1)
        nx.draw(self.multi_graph, pos, node_size=40, node_color=color)

        plt.show()
        return self

def combine_nodes(graph, node_1, node_2, att):
    node_1[att].append(node_2[att])
    for neighbor in list(graph.neighbors(node_2)):
        graph.add_edge(node_1, neighbor)
    
    graph.remove_node(node_2)
    return graph

def combine_nodes_win(graph, node_1, node_2, att):
    
    for i in range(len(list(node_1.nodes(data=True)))):
        for j in range(len(list(node_2.nodes(data=True))[i][1][att])):
            list(node_1.nodes(data=True))[i][1][att].append(list(node_2.nodes(data=True))[i][1][att][j])
    
    for neighbor in list(graph.neighbors(node_2)):
        graph.add_edge(node_1, neighbor)

    graph.remove_node(node_2)
    return graph



path = os.path.join(os.getcwd(), "apple", "APPLE.csv")
"""
TimeSeries().from_csv(CsvStock(path, "Close"))\
    .process(Segment(60, 120))\
    .to_graph(NaturalVisibility())\
    .combine_identical_nodes()\
    .add_edge(0,2)\
    .add_edge(13,35, 17)\
    .draw("blue")


x = TimeSeries().from_csv(CsvStock(path, "Close"))\
    .process(Segment(60, 70))\
    .process(SlidingWindow(5))\
    .to_graph(NaturalVisibility())\
    .draw()

y = TimeSeries().from_csv(CsvStock(path, "Close"))\
    .process(Segment(120, 130))\
    .process(SlidingWindow(5))\
    .to_graph(NaturalVisibility())\
    .draw()

z = TimeSeries().from_csv(CsvStock(path, "Close"))\
    .process(Segment(180, 190))\
    .process(SlidingWindow(5))\
    .to_graph(NaturalVisibility())\
    .draw()

w = TimeSeries().from_csv(CsvStock(path, "Close"))\
    .process(Segment(240, 250))\
    .process(SlidingWindow(5))\
    .to_graph(NaturalVisibility())\
    .draw()

MultivariateTimeSeries()\
    .add(x)\
    .add(y)\
    .add(z)\
    .add(w)\
    .link(Link(multi=True).time_coocurence())\
    .draw("red")


a = MultivariateTimeSeries()\
    .add(x)\
    .add(y)\
    .add(z)\
    .add(w)\
    .link(Link(multi=True).time_coocurence())\
    .combine_identical_nodes()

TimeSeries()\
    .from_csv(CsvStock(path, "Close"))\
    .process(Segment(60, 120))\
    .process(SlidingWindow(5))\
    .to_graph(NaturalVisibility().with_limit(1))\
    .combine_identical_nodes()\
    .draw("pink")

GraphSlidWin(a.return_graph())\
    .set_nodes(a.get_graph_nodes())\
    .walk_through_all()\
    .change_graphs_every_x_steps(2)\
    .choose_next_node("weighted")\
    .choose_next_value("random")\
    .skip_every_x_steps(1)\
    .ts_length(50)\
    .to_multiple_time_sequences()\
    .draw()

GraphSlidWin(x.return_graph())\
    .set_nodes(x.return_graph().nodes)\
    .choose_next_node("weighted")\
    .choose_next_value("random")\
    .skip_every_x_steps(1)\
    .ts_length(50)\
    .to_time_sequence()\
    .draw()
"""
x = TimeSeries().from_csv(CsvStock(path, "Close"))\
    .process(Segment(60, 70))\
    .to_graph(NaturalVisibility())\
    .link(Link().same_timesteps(2))\
    .draw()

y = TimeSeries().from_csv(CsvStock(path, "Close"))\
    .process(Segment(120, 130))\
    .to_graph(NaturalVisibility())\
    .draw()

z = TimeSeries().from_csv(CsvStock(path, "Close"))\
    .process(Segment(180, 190))\
    .to_graph(NaturalVisibility())\
    .draw()

w = TimeSeries().from_csv(CsvStock(path, "Close"))\
    .process(Segment(240, 250))\
    .to_graph(NaturalVisibility())\
    .draw()

a = MultivariateTimeSeries()\
    .add(x)\
    .add(y)\
    .add(z)\
    .add(w)\
    .link(Link().time_coocurence())\
    .combine_identical_nodes()\
    .draw("purple")

Graph(a.return_graph())\
    .set_nodes(a.get_graph_nodes(), a.get_graph_nodes_data())\
    .walk_through_all()\
    .change_graphs_every_x_steps(2)\
    .choose_next_node("weighted")\
    .choose_next_value("random")\
    .skip_every_x_steps(1)\
    .ts_length(50)\
    .to_multiple_time_sequences()\
    .draw()

Graph(x.return_graph())\
    .set_nodes(x.return_graph().nodes, x.return_graph().nodes(data=True))\
    .choose_next_node("weighted")\
    .choose_next_value("random")\
    .skip_every_x_steps(1)\
    .ts_length(50)\
    .to_time_sequence()\
    .draw()

#poimenuj vse edge