import numpy as np


#TODO: to be migrated/superseded by strategies_to_time_sequence
#TODO: after migration, delete
class GraphToTimeseriesStrategy:
    def to_sequence(self, graph, sequence_length):
        return None


class RandomNodeNeighbourSequenceGenerationStrategy(GraphToTimeseriesStrategy):
    def to_sequence(self, graph, sequence_length):
        sequence = []
        nodes = [n for n in graph.nodes()]

        while len(sequence) < sequence_length:
            node = np.random.choice(nodes)
            neighbour = np.random.choice([x for x in graph.neighbors(node)])
            sequence = sequence + [graph.nodes[neighbour]['value']]

        return sequence

class RandomDegreeNodeSequenceGenerationStrategy(GraphToTimeseriesStrategy):
    def to_sequence(self, graph, sequence_length):
        sequence = []
        nodes_weighted_tuples = [(n, float(len([x for x in graph.neighbors(n)]))/float(len(graph.nodes()))) for n in graph.nodes()]
        nodes = [n[0] for n in nodes_weighted_tuples]
        node_weights = [n[1] for n in nodes_weighted_tuples]
        if np.min(node_weights)>0:
            node_weights = np.round(np.divide(node_weights, np.min(node_weights)), decimals=4)
        node_weights = np.divide(node_weights, np.sum(node_weights))
        while len(sequence) < sequence_length:
            node = np.random.choice(nodes, p=node_weights)
            sequence = sequence + [graph.nodes[node]['value']]

        return sequence

class RandomWalkSequenceGenerationStrategy(GraphToTimeseriesStrategy):
    def to_sequence(self, graph, sequence_length):
        sequence = []
        nodes = [n for n in graph.nodes()]
        first_node = np.random.choice(nodes)
        node = first_node
        while len(sequence) < sequence_length:
            sequence = sequence + [graph.nodes[node]['value']]
            nodes = [n for n in graph.neighbors(node)]
            if len(nodes) == 0:
                break
            else:
                node = np.random.choice(nodes)

        return sequence

class RandomWalkWithRestartSequenceGenerationStrategy(GraphToTimeseriesStrategy):
    def to_sequence(self, graph, sequence_length):
        sequence = []
        nodes = [n for n in graph.nodes()]
        first_node = np.random.choice(nodes)
        node = first_node
        while len(sequence) < sequence_length:
            sequence = sequence + [graph.nodes[node]['value']]
            if np.random.random() <0.15:
                node = first_node
            else:
                nodes = [n for n in graph.neighbors(node)]
                if len(nodes) == 0:
                    node = first_node
                else:
                    node = np.random.choice(nodes)

        return sequence
