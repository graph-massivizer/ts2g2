import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from strategies.ts2g import  TS2G_STRATEGY_REGISTRY
from core import Timeseries
from pipeline import T2G2Pipeline
from visualization import draw_g
import os
import utils

amazon_path = os.path.join(os.getcwd(), "data/amazon", "AMZN.csv")

ts_data = utils.CsvFile(amazon_path, "Close").from_csv()
  
# ts_data = np.array([1.0, 2.5, 3.8, 2.2, 1.5, 2])

#to_graph strategy
builder_cls = TS2G_STRATEGY_REGISTRY["proximity"] #registry
strategy_ts = builder_cls(method="recurrence").get_strategy() #builder

pipeline = T2G2Pipeline()
embedding = (
    pipeline.load_timeseries(ts_data)
            .preprocess(method="minmax")
            .apply_view_strategy(view_strategy="slice",start_index=60, end_index=90)
            .to_graph(ts2g_strategy=strategy_ts)
            .merge()
            .to_embedding()
            .run()
)

print("Final embedding:", embedding)
graph = pipeline.get_Graph()
print("graph is ", graph)
draw_g(graph)
