import random
import numpy as np

class StrategyNextValueInNode:
    """
    Stores strategy to chose next value in the selected node.
    
    **Attributes:**

    - `skip`: tells us how many values do we skip before appending next one
    """
    
    def __init__(self):
        self.skip = 0
        self.att = 'value'
        self.dictionaries = None

    def append(self, sequence, graph, graph_index, index):
        pass

    def skip_every_x_steps(self, x):
        self.skip = x
        return self

    def get_skip(self):
        return self.skip

    def set_arguments(self, dictionary, att):
        self.dictionaries = dictionary
        self.att = att

    def get_name(self):
        pass


class StrategyNextValueInNodeRandom(StrategyNextValueInNode):
    """Chooses next value in selected node randomly."""
    def __init__(self):
        super().__init__()

    def append(self, sequence, graph, graph_index, index):
        index = random.randint(0, len(graph[1][self.att]) - 1)
        sequence.append(graph[1][self.att][index])
        return sequence

    def get_name(self):
        return "random"


class StrategyNextValueInNodeRandomForSlidingWindow(StrategyNextValueInNode):
    """Chooses next value in selected node randomly in graph made with sliding window mechanism."""
    def __init__(self):
        super().__init__()

    def append(self, sequence, graph, graph_index, index):
        nodes = list(graph.nodes(data = True))
        random.shuffle(nodes)

        for node in nodes:
            index = random.randint(0, len(node[1][self.att]) - 1)
            sequence.append(node[1][self.att][index])
        return sequence

    def get_name(self):
        return "random"


class StrategyNextValueInNodeRoundRobin(StrategyNextValueInNode):
    """Chooses next value in selected node sequentially, in the same order as they were saved."""
    def __init__(self):
        super().__init__()

    def append(self, sequence, graph, graph_index, index):
        if int(self.dictionaries[graph_index][index]/2) >= len(list(graph[1][self.att])):
            self.dictionaries[graph_index][index] = 0

        ind = int(self.dictionaries[graph_index][index]/2)
        sequence.append(graph[1][self.att][ind])
        self.dictionaries[graph_index][index] += 1
        return sequence

    def get_name(self):
        return "round robin"


class StrategyNextValueInNodeRoundRobinForSlidingWindow(StrategyNextValueInNode):
    """Chooses next value in selected node sequentially for graph made with sliding window mechanism, in the same order as they were saved."""
    def __init__(self):
        super().__init__()

    def append(self, sequence, graph, graph_index, index):
        if int(self.dictionaries[graph_index][index]/2) >= len(list(list(graph.nodes(data=True))[0][1][self.att])):
            self.dictionaries[graph_index][index] = 0

        ind = int(self.dictionaries[graph_index][index]/2)

        for node in graph.nodes(data=True):
            sequence.append(node[1][self.att][ind])

        self.dictionaries[graph_index][index] += 1
        return sequence

    def get_name(self):
        return "round robin"





class StrategySelectNextNode:
    """
    Stores strategy to chose next node from the neighbors of the previous node.
    
    **Attributes:**

    - `change_graphs`: tells how long we walk through one graph, before switching to next one
    - `graph`: networkx.Graph object

    """
    
    def __init__(self):
        self.change_graphs = 1
        self.graph = None
        self.nodes = None
        self.dictionaries = None
        self.att = 'value'

    def next_node(self, i, graph_index, nodes, switch, node):
        pass

    def change_graphs_every_x_steps(self, x):
        self.change_graphs = x
        return self

    def get_change(self):
        return self.change_graphs

    def set_arguments(self, graph, nodes, dictionaries, att):
        self.graph = graph
        self.nodes = nodes
        self.dictionaries = dictionaries
        self.att = att

    def get_name(self):
        pass


class StrategySelectNextNodeRandomlyFromNeighboursAcrossGraphs(StrategySelectNextNode):
    """Walks through all graphs in a multivariate graph and chooses next node randomly."""
    def __init__(self):
        super().__init__()

    def next_node(self, i, graph_index, nodes, switch, node):
        index = int((i/switch) % len(nodes))
        neighbors = set(self.graph.neighbors(nodes[index]))

        neighbors = list(set(self.nodes[graph_index]) & neighbors)
        return random.choice(neighbors)

    def get_name(self):
        return "walkthrough all graphs randomly from neighbours"


class StrategySelectNextNodeRandomlyFromNeighboursFromFirstGraph(StrategySelectNextNode):
    """Walks through first graph and chooses next node randomly."""
    def __init__(self):
        super().__init__()

    def next_node(self, i, graph_index, nodes, switch, node):
        neighbors = set(self.graph.neighbors(node))
        neighbors = list(set(self.nodes[graph_index]) & neighbors)

        return random.choice(neighbors)

    def get_name(self):
        return "walkthrough one graph randomly from neighbours"


class StrategySelectNextNodeRandomly(StrategySelectNextNode):
    """Randomly chooses next node from all nodes of the graph."""
    def __init__(self):
        super().__init__()
    
    def next_node(self, i, graph_index, nodes, switch, node):
        return random.choice(self.nodes[graph_index])
    
    def get_name(self):
        return "Random walkthrough the nodes"
    
class StrategySelectNextNodeRandomDegree(StrategySelectNextNode):
    """Randomly chooses next node in graph based on a neighbor degree."""
    def __init__(self):
        super().__init__()

    def next_node(self, i, graph_index, nodes, switch, node):
        nodes_weighted_tuples = [(n, float(len([x for x in list(set(self.nodes[graph_index]) & set(self.graph.neighbors(node)))]))/float(len(nodes[graph_index]))) for n in list(set(self.graph.neighbors(node)) & set(self.nodes[graph_index]))]
        nodes_new = [n[0] for n in nodes_weighted_tuples]
        node_weights = [n[1] for n in nodes_weighted_tuples]
        if np.min(node_weights)>0:
            node_weights = np.round(np.divide(node_weights, np.min(node_weights)), decimals=4)
        node_weights = np.divide(node_weights, np.sum(node_weights))

        numbers = [i for i in range(len(nodes_new))]

        random_choice = np.random.choice(numbers, p=node_weights)
        return nodes_new[random_choice]
    

    def get_name(self):
        return "Random degree walkthrough the nodes"
    

class StrategySelectNextNodeRandomWithRestart(StrategySelectNextNode):
    """Randomly chooses next node with 15% chance of choosing the first node."""
    def __init__(self):
        super().__init__()
        self.first_node = None
    
    def next_node(self, i, graph_index, nodes, switch, node):
        if self.first_node == None:
            self.first_node = []
            for i in range(len(nodes)):
                numbers = [j for j in range(len(self.nodes[i]))]
                random_choice = np.random.choice(numbers)
                self.first_node.append(self.nodes[i][random_choice])
        

        if np.random.random() <0.15:
            return self.first_node[graph_index]
        
        if len(nodes) == 0:
            node = self.first_node[graph_index]
        else:
            numbers = [j for j in range(len(self.nodes[graph_index]))]
            random_choice = np.random.choice(numbers)
            node = self.nodes[graph_index][random_choice]

        return node
    
    def get_name(self):
        return "Random walk with restart"
