
import numpy as np
import hashlib
from core.model import TimeseriesView
from scipy import stats


class VisitorTimeseriesEmbeddingModel:
    def __init__(self, model = None):
        self.model = model
    
    def predict(self, timeseries):
        pass

class VisitorGraphEmbeddingModel:
    def __init__(self, model = None):
        self.model = model
    
    def predict(self, graph):
        pass

class EmbeddingRanking:

    def __init__(self, embedding_length = 20):
        self.dictionaries = None
        self.to_graph_methods = []
        self.timeseries_model = None
        self.graph_model = None
        self.embedding_length = embedding_length
        self.base_vector = [0.5 for i in range(embedding_length)]

    def id(self, timeseries):
        return hashlib.md5(str(timeseries).encode()).hexdigest()
    
    def set_to_graph_strategies(self, array):
        self.to_graph_methods = array
        self.dictionaries = [{} for i in range(len(array)+1)]
        return self
    
    def set_embedding_models(self, timeseries_model: VisitorTimeseriesEmbeddingModel, graph_model: VisitorGraphEmbeddingModel):
        self.timeseries_model = timeseries_model
        self.graph_model = graph_model
        return self
    
    def add_timeseries(self, timeseries: TimeseriesView):
        ts = timeseries.get_ts()
        ts_id = self.id(ts)
        self.dictionaries[0][ts_id] = self.timeseries_model.predict(ts)
        for i in range(len(self.to_graph_methods)):
            self.dictionaries[i+1][ts_id] = self.graph_model.predict(timeseries.to_graph(self.to_graph_methods[i].get_strategy())._get_graph())
        return self
    
    def embedding_ranking(self):
        self.ranking = []

        for stage in self.dictionaries:
            ids = list(stage.keys())
            embeddings = [stage[ids[i]] for i in range(len(ids))]
            distances = []
            for vector in embeddings:
                distances.append(self.cosine_distance(vector))
            
            sorted_pairs = sorted(zip(ids, distances))
            sorted_ids, sorted_distances = zip(*sorted_pairs)
            sorted_ids = list(sorted_ids)
            self.ranking.append(sorted_ids)

        return self 
    
    def kendall_tau_correlation(self):
        correlation = []
        for i in range(len(self.to_graph_methods)):
            correlation.append(stats.kendalltau(self.ranking[0], self.ranking[i+1]).statistic)
        for i in range(len(self.to_graph_methods)):
            print(f"{self.to_graph_methods[i].get_strategy()._get_name()}: {correlation[i]}")
        return self
    
    def cosine_distance(self, vector):
        print(vector)
        print(len(vector[0]), len(vector))
        print(self.base_vector)
        dot_product = np.dot(self.base_vector, vector)
        norm_1 = np.linalg.norm(self.base_vector)
        norm_2 = np.linalg.norm(vector)
        cosine_similarity = dot_product / (norm_1*norm_2)
        return 1 - cosine_similarity



"""
def get_euclidean_distance(self, time_graph_1: TimeGraph, time_graph_2: TimeGraph):
    hash_1 = time_graph_1._hash()
    hash_2 = time_graph_2._hash()
    vector_1 = self.embeddings[hash_1]
    vector_2 = self.embeddings[hash_2]
    distance = 0
    for i in range(len(vector_1)):
        distance += (vector_1[i]-vector_2[i])*(vector_1[i]-vector_2[i])
    distance = np.sqrt(distance)
    print(distance)
    return self

def rbo(self, time_graph_1: TimeGraph, time_graph_2: TimeGraph, p=0.9):
    hash_1 = time_graph_1._hash()
    hash_2 = time_graph_2._hash()
    list1 = self.embeddings[hash_1]
    list2 = self.embeddings[hash_2]

    # tail recursive helper function
    def helper(ret, i, d):
        l1 = set(list1[:i]) if i < len(list1) else set(list1)
        l2 = set(list2[:i]) if i < len(list2) else set(list2)
        a_d = len(l1.intersection(l2))/i
        term = math.pow(p, i) * a_d
        if d == i:
            return ret + term
        return helper(ret + term, i + 1, d)
    k = max(len(list1), len(list2))
    x_k = len(set(list1).intersection(set(list2)))
    summation = helper(0, 1, k)
    return ((float(x_k)/k) * math.pow(p, k)) + ((1-p)/p * summation)
    """