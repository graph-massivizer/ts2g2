from embeddings.ts2g2_embeddings import TrainGraphEmbeddingModel
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import networkx as nx

# ========= FUNCTIONS GET RANDOM WALKS, TRAIN =======================================================

# Function that trains on already created wolks, thus it takes them as input and returns model
def train_graph_embedding_model(rand_walks, embedding_size: int = 20):

    walks = [rand_walks]
    documents_gensim = []
    for i, doc_walks in enumerate(walks):
            for doc_walk in doc_walks:
                    documents_gensim.append(TaggedDocument(doc_walk, [i]))
            
    model = Doc2Vec(documents_gensim, vector_size=embedding_size, window=3, min_count=1, workers=4)
    model.train(documents_gensim, total_examples=model.corpus_count, epochs=model.epochs)

    return model

# Graph input type: core.model.TimeGraph
def get_random_walks_for_graph(graph):
        graph = graph._get_graph()
        df = pd.DataFrame(graph.edges(data=True), columns = ['source', 'target', 'attributes'])
        G = nx.from_pandas_edgelist(df, 'source', 'target')
        walks = nx.generate_random_paths(G, sample_size=15, path_length=45)
        str_walks = [[str(n) for n in walk] for walk in walks]

        return str_walks

# ========= FUNCTIONS USED FOR DATAFRAMES CONTAINING GRAPHS =======================================================
# Fuction that can be applied to a dataframe of graphs (it contains a column with graphs(one graph in each row))
# It makes a new column with random walks for each graph
def get_random_walks_from_graph_df(df_graph):
    
    # Create a new column to store random walks for each graph
    random_walks_column = []
    
    for idx, row in df_graph.iterrows():
        graph = row['Graph']  # Replace with the actual name of the graph column, if different
        graph= graph._get_graph()
        if(len(graph.edges()) > 0):
            walks = get_random_walks_for_graph(graph)
            random_walks_column.append(walks)
        else:
            random_walks_column.append(np.nan)

    # Add the new random walks column to the DataFrame
    df_graph['Random_Walks'] = random_walks_column
    df_random_walks = df_graph[['UUID', 'Graph']] 
    df_rand_walk = df_rand_walk.dropna(subset=['Random_Walks'])
    
    return df_random_walks


def train_graph_embedding_model_df(df_rand_walk, embedding_size: int = 20):
    
    df_rand_walk['Doc2Vec_Model'] = None  # Initialize a new column

    for idx, row in df_rand_walk.iterrows():

            model = train_graph_embedding_model(row['Random_Walks'], embedding_size)
            df_rand_walk.at[idx, 'Doc2Vec_Model'] = model
    
    df_model = df_rand_walk
            
    return df_model

