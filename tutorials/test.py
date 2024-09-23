import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from core.model import Timeseries, TimeseriesPreprocessing, TimeseriesPreprocessingSegmentation, TimeseriesPreprocessingSlidingWindow, TimeseriesPreprocessingComposite, TimeseriesView, TimeGraph, ToSequenceVisitorSlidingWindow, ToSequenceVisitor, ToSequenceVisitorOrdinalPartition
from tsg_io.input import CsvFile, TsFile
from from_graph.strategy_to_time_sequence import StrategyNextValueInNodeRandom, StrategyNextValueInNodeRandomForSlidingWindow, StrategyNextValueInNodeRoundRobin, StrategyNextValueInNodeRoundRobinForSlidingWindow, StrategySelectNextNodeRandomlyFromNeighboursAcrossGraphs, StrategySelectNextNodeRandomlyFromNeighboursFromFirstGraph, StrategySelectNextNodeRandomly, StrategySelectNextNodeRandomDegree, StrategySelectNextNodeRandomWithRestart, StrategyNextValueInNodeOrdinalPartition
from to_graph.strategy_linking_graph import StrategyLinkingGraphByValueWithinRange, LinkNodesWithinGraph
from to_graph.strategy_linking_multi_graphs import LinkGraphs
from to_graph.strategy_to_graph import BuildTimeseriesToGraphNaturalVisibilityStrategy, BuildTimeseriesToGraphHorizontalVisibilityStrategy, BuildTimeseriesToGraphOrdinalPartition, BuildTimeseriesToGraphQuantile
from embeddings.ts2g2_embeddings import EmbeddingRanking, VisitorGraphEmbeddingModelDoc2Vec, VisitorTimeseriesEmbeddingModelTS2Vec
import numpy as np
import networkx as nx
import joblib
import pandas as pd

amazon_path = os.path.join(os.getcwd(), "amazon", "AMZN.csv")
apple_path = os.path.join(os.getcwd(), "apple", "APPLE.csv")
abnormalHeartbeat_path = os.path.join(os.getcwd(), "abnormal_heartbeat", "AbnormalHeartbeat_TEST.ts")
acsf1_path = os.path.join(os.getcwd(), "ACSF1", "ACSF1_TEST.ts")
adiac_path = os.path.join(os.getcwd(), "adiac", "Adiac_TEST.ts")
dodger_loop_weekend_path = os.path.join(os.getcwd(), "dodger_loop_weekend", "DodgerLoopWeekend_TEST.ts")




timegraph_1 = Timeseries(TsFile(abnormalHeartbeat_path).from_ts())\
    .with_preprocessing(TimeseriesPreprocessingSegmentation(60, 90))\
    .to_graph(BuildTimeseriesToGraphNaturalVisibilityStrategy().with_limit(1).get_strategy())\
    .add_edge(0,2)\
    .add_edge(13, 21, weight = 17)\
    .link(LinkNodesWithinGraph().by_value(StrategyLinkingGraphByValueWithinRange(2)).seasonalities(15))\
    #.draw("blue")

timegraph_ordinal_partition = Timeseries(TsFile(abnormalHeartbeat_path).from_ts())\
    .with_preprocessing(TimeseriesPreprocessingSegmentation(60, 120))\
    .add(Timeseries(TsFile(abnormalHeartbeat_path).from_ts())\
        .with_preprocessing(TimeseriesPreprocessingSegmentation(120, 180)))\
    .add(Timeseries(TsFile(abnormalHeartbeat_path).from_ts())\
        .with_preprocessing(TimeseriesPreprocessingSegmentation(500, 560)))\
    .add(Timeseries(TsFile(abnormalHeartbeat_path).from_ts())\
        .with_preprocessing(TimeseriesPreprocessingSegmentation(700, 760)))\
    .add(Timeseries(CsvFile(amazon_path, "Close").from_csv())\
        .with_preprocessing(TimeseriesPreprocessingSegmentation(1000, 1060)))\
    .to_histogram(15)\
    .to_graph(BuildTimeseriesToGraphOrdinalPartition(10, 5).get_strategy())\
    .link(LinkGraphs().time_cooccurrence())\
    .add_edge(0,2)\
    .link(LinkNodesWithinGraph().seasonalities(4))\
    #.draw("purple")

"""
timegraph_ordinal_partition.to_sequence(ToSequenceVisitorOrdinalPartition()\
    .next_node_strategy(StrategySelectNextNodeRandomWithRestart())\
    .next_value_strategy(StrategyNextValueInNodeOrdinalPartition())\
    .ts_length(100))\
    #.draw_sequence()
"""


timegraph_quantile = Timeseries(CsvFile(amazon_path, "Close").from_csv())\
    .with_preprocessing(TimeseriesPreprocessingSegmentation(60, 120))\
    .to_graph(BuildTimeseriesToGraphQuantile(4, 1).get_strategy())\
    .add_edge(0,2)\
    .link(LinkNodesWithinGraph().seasonalities(4))\
    #.draw("grey")


timegraph_2 = Timeseries(CsvFile(apple_path, "Close").from_csv())\
    .with_preprocessing(TimeseriesPreprocessingComposite()\
        .add(TimeseriesPreprocessingSegmentation(60, 120))\
        .add(TimeseriesPreprocessingSlidingWindow(5)))\
    .to_graph(BuildTimeseriesToGraphNaturalVisibilityStrategy().get_strategy())\
    .link(LinkGraphs().sliding_window())\
    .combine_identical_subgraphs()\
    #.draw("red")


timegraph_3 = Timeseries(CsvFile(apple_path, "Close").from_csv())\
    .with_preprocessing(TimeseriesPreprocessingSegmentation(60, 90))\
    .add(Timeseries(CsvFile(apple_path, "Close").from_csv())\
        .with_preprocessing(TimeseriesPreprocessingSegmentation(90, 120)))\
    .add(Timeseries(CsvFile(apple_path, "Close").from_csv())\
        .with_preprocessing(TimeseriesPreprocessingSegmentation(150, 180)))\
    .to_graph(BuildTimeseriesToGraphNaturalVisibilityStrategy().with_limit(1).get_strategy())\
    .link(LinkGraphs().time_cooccurrence())\
    .link(LinkNodesWithinGraph().by_value(StrategyLinkingGraphByValueWithinRange(0.5)))\
    .combine_identical_nodes()\
    #.draw("brown")


timegraph_4 = Timeseries(CsvFile(apple_path, "Close").from_csv())\
    .with_preprocessing(TimeseriesPreprocessingComposite()\
        .add(TimeseriesPreprocessingSegmentation(60, 110))\
        .add(TimeseriesPreprocessingSlidingWindow(5)))\
    .add(Timeseries(CsvFile(apple_path, "Close").from_csv())\
        .with_preprocessing(TimeseriesPreprocessingComposite()\
            .add(TimeseriesPreprocessingSegmentation(120, 170))\
            .add(TimeseriesPreprocessingSlidingWindow(5)))\
        .add(Timeseries(CsvFile(apple_path, "Close").from_csv())\
            .with_preprocessing(TimeseriesPreprocessingComposite()\
                    .add(TimeseriesPreprocessingSegmentation(190, 240))\
                    .add(TimeseriesPreprocessingSlidingWindow(5)))))\
    .to_graph(BuildTimeseriesToGraphNaturalVisibilityStrategy().get_strategy())\
    .link(LinkGraphs().sliding_window().time_cooccurrence())\
    .combine_identical_subgraphs()\
    .link(LinkNodesWithinGraph().seasonalities(15))\
    #.draw("green")



timegraph_6 = Timeseries(CsvFile(amazon_path, "Close").from_csv())\
    .with_preprocessing(TimeseriesPreprocessingSegmentation(4000, 4500))\
    .to_graph(BuildTimeseriesToGraphNaturalVisibilityStrategy().with_limit(1).get_strategy())\

timegraph_7 = Timeseries(CsvFile(amazon_path, "Close").from_csv())\
    .with_preprocessing(TimeseriesPreprocessingSegmentation(2000, 2100))\
    .to_graph(BuildTimeseriesToGraphHorizontalVisibilityStrategy().with_limit(1).get_strategy())\

timegraph_8 = Timeseries(CsvFile(amazon_path, "Close").from_csv())\
    .with_preprocessing(TimeseriesPreprocessingComposite()\
        .add(TimeseriesPreprocessingSegmentation(950, 1050))\
        .add(TimeseriesPreprocessingSlidingWindow(10)))\
    .to_graph(BuildTimeseriesToGraphHorizontalVisibilityStrategy().with_limit(1).get_strategy())\
    .link(LinkGraphs().sliding_window())\
    .combine_identical_subgraphs()\

path = TsFile(abnormalHeartbeat_path).from_ts()


model_graph = VisitorGraphEmbeddingModelDoc2Vec().train_model([timegraph_1, timegraph_2, timegraph_3, timegraph_4, timegraph_6, timegraph_7, timegraph_8, timegraph_ordinal_partition, timegraph_quantile], 20)
model_ts = VisitorTimeseriesEmbeddingModelTS2Vec().train_model(path, 20, epoch=20)





joblib.dump(model_graph, "embedding_models/graph_model_abnormal_heartbeat.joblib")
joblib.dump(model_ts, "embedding_models/timeseries_model_abnormal_heartbeat.joblib")


"""
model_graph = joblib.load("embedding_models/graph_model.joblib")
model_ts = joblib.load("embedding_models/timeseries_model.joblib")
"""




data = {'run':[], 'natural_visibility':[], 'horizontal_visibility':[], 'ordinal_partition':[], 'quantile':[]}
    
i = 1
while i <= 5:
    print(i)

    x = EmbeddingRanking(20)\
        .set_embedding_models(model_ts, model_graph)\
        .set_to_graph_strategies([BuildTimeseriesToGraphNaturalVisibilityStrategy(), BuildTimeseriesToGraphHorizontalVisibilityStrategy(), BuildTimeseriesToGraphOrdinalPartition(10, 5), BuildTimeseriesToGraphQuantile(4, 1)])\
        .add_timeseries(Timeseries(path).with_preprocessing(TimeseriesPreprocessingSegmentation(100, 200)))\
        .add_timeseries(Timeseries(path).with_preprocessing(TimeseriesPreprocessingSegmentation(300, 400)))\
        .add_timeseries(Timeseries(path).with_preprocessing(TimeseriesPreprocessingSegmentation(500, 600)))\
        .add_timeseries(Timeseries(path).with_preprocessing(TimeseriesPreprocessingSegmentation(700, 800)))\
        .add_timeseries(Timeseries(path).with_preprocessing(TimeseriesPreprocessingSegmentation(900, 1000)))\
        .add_timeseries(Timeseries(path).with_preprocessing(TimeseriesPreprocessingSegmentation(200, 350)))\
        .add_timeseries(Timeseries(path).with_preprocessing(TimeseriesPreprocessingSegmentation(550, 680)))\
        .add_timeseries(Timeseries(path).with_preprocessing(TimeseriesPreprocessingSegmentation(30, 80)))\
        .add_timeseries(Timeseries(path).with_preprocessing(TimeseriesPreprocessingSegmentation(0, 300)))\
        .add_timeseries(Timeseries(path).with_preprocessing(TimeseriesPreprocessingSegmentation(1100, 1250)))\
        .embedding_ranking()\
        .kendall_tau_correlation()
    data['run'].append(i)
    i+=1
    data['natural_visibility'].append(x[0])
    data["horizontal_visibility"].append(x[1])
    data["ordinal_partition"].append(x[2])
    data["quantile"].append(x[3])

df = pd.DataFrame.from_dict(data)
average_values = df.mean()
print(average_values)
df.to_csv('kendall_tau_results/apple_kendall_tau', index=False)






"""

timegraph_1.to_sequence(ToSequenceVisitor()\
        .next_node_strategy(StrategySelectNextNodeRandomWithRestart())\
        .next_value_strategy(StrategyNextValueInNodeRoundRobin().skip_every_x_steps(1))\
        .ts_length(100))\
    .draw_sequence()


timegraph_2.to_sequence(ToSequenceVisitorSlidingWindow()\
    .next_node_strategy(StrategySelectNextNodeRandomWithRestart())\
    .next_value_strategy(StrategyNextValueInNodeRandomForSlidingWindow().skip_every_x_steps(1))\
    .ts_length(50))\
    .draw_sequence()


timegraph_3.to_sequence(ToSequenceVisitor()\
    .next_node_strategy(StrategySelectNextNodeRandomWithRestart().change_graphs_every_x_steps(2))\
    .next_value_strategy(StrategyNextValueInNodeRoundRobin().skip_every_x_steps(1))\
    .ts_length(50))\
    .draw_sequence()


timegraph_4.to_sequence(ToSequenceVisitorSlidingWindow()\
    .next_node_strategy(StrategySelectNextNodeRandomWithRestart())\
    .next_value_strategy(StrategyNextValueInNodeRoundRobinForSlidingWindow())\
    .ts_length(100))\
    .draw_sequence()


"""
"""

import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

import os
import csv
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from timeseries.strategies import TimeseriesToGraphStrategy, TimeseriesEdgeVisibilityConstraintsNatural, TimeseriesEdgeVisibilityConstraintsHorizontal, EdgeWeightingStrategyNull
from core import model 
from sklearn.model_selection import train_test_split
from tsg_io.input import CsvFile

apple_data = pd.read_csv(os.path.join(os.getcwd(), "apple", "APPLE.csv"))

timegraph_1 = model.Timeseries(CsvFile(os.path.join(os.getcwd(), "apple", "APPLE.csv"), "Close").from_csv()).get_ts()


def plot_timeseries(sequence, title, x_legend, y_legend, color):
    plt.figure(figsize=(10, 6))
    plt.plot(sequence, linestyle='-', color=color)
    
    plt.title(title)
    plt.xlabel(x_legend)
    plt.ylabel(y_legend)
    plt.grid(True)
    plt.show()


def plot_timeseries_sequence(df_column, title, x_legend, y_legend, color='black'):
    sequence = model.Timeseries(model.TimeseriesArrayStream(df_column)).to_sequence()
    plot_timeseries(sequence, title, x_legend, y_legend, color)


segment_1 = timegraph_1[60:110]
segment_2 = timegraph_1[120:170]
segment_3 = timegraph_1[190:240]

plot_timeseries(segment_1, "", "Date", "Value", "black")

plot_timeseries(segment_2, "", "Date", "Value", "black")

plot_timeseries(segment_3, "", "Date", "Value", "black")
"""
