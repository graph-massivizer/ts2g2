
import numpy as np
import hashlib
from core.model import TimeseriesView
from scipy import stats


class VisitorTimeseriesEmbeddingModel:
    def __init__(self):
        self.model = None
    
    def predict(self, timeseries):
        pass

class TrainModel:
    def __init__(self):
        self.model = None

    def get_model(self):
        pass

class VisitorGraphEmbeddingModel:
    def __init__(self, model):
        self.model = None
    
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
        self.poskus = 0
        self.abeceda = "abcdefghijklmonpqrstuvwxyz"

    def id(self, timeseries):
        x = self.abeceda[self.poskus]
        self.poskus += 1
        return x
        #return hashlib.md5(str(timeseries).encode()).hexdigest()
    
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
            self.dictionaries[i+1][ts_id] = self.graph_model.predict(timeseries.to_graph(self.to_graph_methods[i].get_strategy()))
        return self
    
    def embedding_ranking(self):
        self.ranking = []

        for stage in self.dictionaries:
            ids = list(stage.keys())
            embeddings = [stage[ids[i]] for i in range(len(ids))]
            distances = []
            for vector in embeddings:
                distances.append(self.cosine_distance(vector))
            
            sorted_pairs = sorted(zip(distances, ids))
            sorted_distances, sorted_ids = zip(*sorted_pairs)
            sorted_ids = list(sorted_ids)
            self.ranking.append(sorted_ids)
        """
        k = 1
        for j in range(len(self.ranking[0])):
            print(f"{k}:", end = " ")
            k+=1
            for i in range(len(self.ranking)):
                print(f"{self.ranking[i][j]}", end = " | ")
            print()
        """
                
        return self 
    
    def kendall_tau_correlation(self):
        correlation = []
        for i in range(len(self.to_graph_methods)):
            correlation.append(stats.kendalltau(self.ranking[0], self.ranking[i+1]).statistic)
        """for i in range(len(self.to_graph_methods)):
            print(f"{self.to_graph_methods[i].get_strategy()._get_name()}: {correlation[i]}")"""
        return correlation
    
    def cosine_distance(self, vector):
        dot_product = np.dot(self.base_vector, vector)
        norm_1 = np.linalg.norm(self.base_vector)
        norm_2 = np.linalg.norm(vector)
        cosine_similarity = dot_product / (norm_1*norm_2)
        return 1 - cosine_similarity



import networkx as nx
import pandas as pd
import numba
import json
from numpy import triu
from scipy.linalg import get_blas_funcs
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Kudos to the following pages, for clarifications:
# https://stackoverflow.com/questions/66665981/should-i-split-sentences-in-a-document-for-doc2vec
# https://medium.com/@klintcho/doc2vec-tutorial-using-gensim-ab3ac03d3a1



from embeddings.ts2vec.ts2vec import TS2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

class TrainGraphEmbeddingModel(TrainModel):
    def __init__(self):
        self.model = None
    
    def get_random_walks_for_graph(self, df_graph):
        df = pd.DataFrame(df_graph.edges(data=True), columns = ['source', 'target', 'attributes'])
        G = nx.from_pandas_edgelist(df, 'source', 'target')
        walks = nx.generate_random_paths(G, sample_size=15, path_length=45)

        str_walks = [[str(n) for n in walk] for walk in walks]
        return str_walks

    def train_model(self, graphs, embedding_size):
        documents = []
        for idx in range(len(graphs)):
            source_target_dataframe = graphs[idx]._get_graph()
            document = self.get_random_walks_for_graph(source_target_dataframe)
            documents = documents + [document]

        documents_gensim = []
        for i, doc_walks in enumerate(documents):
            for doc_walk in doc_walks:
                documents_gensim = documents_gensim + [TaggedDocument(doc_walk, [i])]

        model = Doc2Vec(documents_gensim, vector_size=embedding_size, window=3, min_count=1, workers=4)

        model.train(documents_gensim, total_examples=model.corpus_count, epochs=model.epochs)
        self.model = model
        return self
    
    def get_model(self):
        return self.model


class TrainTimeseriesEmbeddingModel(TrainModel):
    def __init__(self):
        self.model = None


    def train_model(self, timeseries, embedding_size, epoch = None):
        train_data, test_data= train_test_split(timeseries, random_state=42)

        scaler = StandardScaler()
        if(len(train_data.shape) == 1):
            train_data = train_data.to_frame()
            train_data = scaler.fit_transform(train_data)

        if(len(train_data.shape) == 2):
            train_data = train_data.reshape(1, train_data.shape[0], train_data.shape[1])

        self.model = TS2Vec(input_dims = 1, output_dims = embedding_size, device='cpu')

        self.model.fit(train_data, n_epochs = epoch, verbose=True)
        train_embeddings = self.model.encode(train_data)
        return self
    
    def get_model(self):
        return self.model
    
class VisitorGraphEmbeddingModelDoc2Vec(VisitorGraphEmbeddingModel):
    def __init__(self, model):
        self.model = model
    
    def get_random_walks_for_graph(self, df_graph):
        df = pd.DataFrame(df_graph.edges(data=True), columns = ['source', 'target', 'attributes'])
        G = nx.from_pandas_edgelist(df, 'source', 'target')
        walks = nx.generate_random_paths(G, sample_size=15, path_length=45)
        str_walks = [[str(n) for n in walk] for walk in walks]
        return str_walks
    
    def predict(self, graph):
        doc = self.get_random_walks_for_graph(graph._get_graph())
        documents_gensim = []
        for i, doc_walks in enumerate(doc):
            documents_gensim = documents_gensim + [''.join(TaggedDocument(doc_walks, [i]).words)]
        return self.model.infer_vector(documents_gensim)
        
class VisitorTimeseriesEmbeddingModelTS2Vec(VisitorTimeseriesEmbeddingModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, timeseries):
        while(isinstance(timeseries, list)):
            timeseries = timeseries[0]
        if(len(timeseries.shape) == 1):
            timeseries = timeseries.to_frame()
            scaler = StandardScaler()
            timeseries = scaler.fit_transform(timeseries)
        
        if(len(timeseries.shape) == 2):
            timeseries = timeseries.reshape(1, timeseries.shape[0], timeseries.shape[1])
        x = self.model.encode(timeseries, encoding_window='full_series')
        return x[0]
    
    






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