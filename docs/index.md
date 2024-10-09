# ts2g<sup>2</sup>

TS2G<sup>2</sup> stands for "timeseries to graphs and back". The library implements a variety of strategies to convert timeseries into graphs, and convert graphs into sequences. Below, we provide a code snippet to generate a graph from timeseries:
    
    # load time series from a file
    timegraph = model.Timeseries(CsvStock(some_file_path, "ColumnOfInterest").from_csv())\

    # and preprocess the timeseries with multiple preprocessing strategies
    .with_preprocessing(model.TimeseriesPreprocessingComposite()\
        .add_strategy(model.TimeseriesPreprocessingSegmentation(60, 120))\
        .add_strategy(model.TimeseriesPreprocessingSlidingWindow(5)))\

    # then create a graph from the timeseries, following a particular strategy
    .to_graph(tgs.BuildTimeseriesToGraphNaturalVisibilityStrategy().get_strategy())\

    # link graphs that result from the same timeseries, but at different sliding window frames
    .link(mgl.LinkGraphs().sliding_window())\

    # and combine identical graphs that result from the abovementioned time windows into single nodes
    .combine_identical_nodes_slid_win()\

    # finally, draw the graph
    .draw("red")

For a more detailed example, look at the [Amazon stocks demo](https://github.com/graph-massivizer/ts2g2/blob/main/tutorials/demo-ts2g2.ipynb).

Many of the methods implemented in this library are described in _Silva, Vanessa Freitas, et al. "Time series analysis via network science: Concepts and algorithms." Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery 11.3 (2021): e1404._ Nevertheless, the library also includes additional techniques found in other works from the scientific literature.

This package is being developed as part of the [Graph-Massivizer](https://graph-massivizer.eu/) project. 
The package is a joint effort between the [Jožef Stefan Institute](https://www.ijs.si/), the [University of Twente](https://www.utwente.nl/en/), the [Vrije Universiteit Amsterdam](https://vu.nl/en), the [University of Klagenfurt](https://www.aau.at/en/), the [University of Bologna](https://www.unibo.it/en), and [Peracton](https://peracton.com/).


### Converting timeseries into graphs

#### Implemented features

<table class="tg">
<thead>
  <tr>
    <th class="tg-7btt" rowspan="3">#</th>
    <th class="tg-7btt" rowspan="3">Visibility Graph</th>
    <th class="tg-7btt" colspan="3">Graph type</th>
    <th class="tg-7btt" colspan="4">Constraints</th>
  </tr>
  <tr>
    <th class="tg-7btt" rowspan="2">Undirected</th>
    <th class="tg-7btt" rowspan="2">Directed</th>
    <th class="tg-7btt" rowspan="2">Weighted</th>
  </tr>
  <tr>
    <th class="tg-7btt">Penetration</th>
    <th class="tg-7btt">Angle</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7btt">1</td>
    <td class="tg-0pky">Natural Visibility Graph</td>
    <td class="tg-0pky">X</td>
    <td class="tg-0pky">X</td>
    <td class="tg-0pky">X</td>
    <td class="tg-0pky">X</td>
    <td class="tg-0pky">X</td>
  </tr>
  <tr>
    <td class="tg-7btt">2</td>
    <td class="tg-0pky">Horizontal Visibility Graph</td>
    <td class="tg-0pky">X</td>
    <td class="tg-0pky">X</td>
    <td class="tg-0pky">X</td>
    <td class="tg-0pky">X</td>
    <td class="tg-0pky">X</td>
  </tr>
  <tr>
    <td class="tg-7btt">3</td>
    <td class="tg-0pky">Difference Visibility Graph</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-7btt">4</td>
    <td class="tg-0pky">Quantile Graph</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">X</td>
    <td class="tg-0pky">X</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
  </tr>

  <tr>
    <td class="tg-7btt">5</td>
    <td class="tg-0pky">Ordinal Partition Graph</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">X</td>
    <td class="tg-0pky">X</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
  </tr>
</tbody>
</table>

#### References table

<table class="tg">
<thead>
  <tr>
    <th class="tg-7btt" rowspan="3">#</th>
    <th class="tg-7btt" rowspan="3">Visibility Graph</th>
    <th class="tg-7btt" colspan="3">Graph type</th>
    <th class="tg-7btt" colspan="4">Constraints</th>
  </tr>
  <tr>
    <th class="tg-7btt" rowspan="2">Undirected</th>
    <th class="tg-7btt" rowspan="2">Directed</th>
    <th class="tg-7btt" rowspan="2">Weighted</th>
  </tr>
  <tr>
    <th class="tg-7btt">Penetration</th>
    <th class="tg-7btt">Angle</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7btt">1</td>
    <td class="tg-0pky">Natural Visibility Graph</td>
    <td class="tg-0pky">
        <a href="https://www.pnas.org/doi/10.1073/pnas.0709247105">ref</a>
    </td>
    <td class="tg-0pky">
      <!-- directed -->
      <a href="https://link.springer.com/article/10.1140/epjb/e2012-20809-8">ref</a>
    </td>
    <td class="tg-0pky">
      <!-- weighted -->
    </td>
    <td class="tg-0pky">
      <!-- constraints:references: penetration -->
      <a href="https://www.semanticscholar.org/paper/Limited-penetrable-visibility-graph-for-complex-Zhou-Jin/fe4a3d2f486021ee066f8a80472deef57d8aee71">ref</a>
    </td>
    <td class="tg-0pky">
      <!-- constraints:references: angle -->
      <a href="https://doi.org/10.1109/ACCESS.2016.2612242">ref</a>, 
      <a href="https://doi.org/10.1016/j.physa.2014.07.002">ref</a>
    </td>
  </tr>
  <tr>
    <td class="tg-7btt">2</td>
    <td class="tg-0pky">Horizontal Visibility Graph</td>
    <td class="tg-0pky">
      <a href="https://journals.aps.org/pre/abstract/10.1103/PhysRevE.80.046103">ref</a>
    </td>
    <td class="tg-0pky">
      <a href="https://link.springer.com/article/10.1140/epjb/e2012-20809-8">ref</a>
      <!-- directed -->
    </td>
    <td class="tg-0pky">
      <!-- weighted -->
      <a href="https://doi.org/10.1109/ACCESS.2016.2612242">ref</a> 
    </td>
    <td class="tg-0pky">
      <!-- constraints:references: penetration -->
      <a href="https://www.semanticscholar.org/paper/Limited-penetrable-visibility-graph-for-complex-Zhou-Jin/fe4a3d2f486021ee066f8a80472deef57d8aee71">ref</a>, 
      <a href="https://www.nature.com/articles/srep35622">ref</a>
    </td>
    <td class="tg-0pky">
      <!-- constraints:references: angle -->
    </td>
  </tr>
  <tr>
    <td class="tg-7btt">3</td>
    <td class="tg-0pky">Difference Visibility Graph</td>
    <td class="tg-0pky">
        <!-- undirected -->
    </td>
    <td class="tg-0pky">
      <!-- directed -->
    </td>
    <td class="tg-0pky">
      <!-- weighted -->
    </td>
    <td class="tg-0pky">
      <!-- constraints:references: penetration -->
    </td>
    <td class="tg-0pky">
      <!-- constraints:references: angle -->
    </td>
  </tr>
<tr>
    <td class="tg-7btt">4</td>
    <td class="tg-0pky">Quantile Graph</td>
    <td class="tg-0pky">
    </a>
    </td>
    <td class="tg-0pky">
      <!-- directed -->
      <a href="https://www.researchgate.net/figure/Illustrative-example-of-the-quantile-graph-algorithm-for-Q-4-On-the-left-panel-we_fig8_349721222">ref</a>
    </td>
    <td class="tg-0pky">
      <!-- weighted -->
    </td>
    <td class="tg-0pky">
      <!-- constraints:references: penetration -->
    </a>
    </td>
    <td class="tg-0pky">
      <!-- constraints:references: angle -->
    </a>
    </td>
  </tr><tr>
    <td class="tg-7btt">5</td>
    <td class="tg-0pky">Ordinal Partition Graph</td>
    <td class="tg-0pky">
    </a>
    </td>
    <td class="tg-0pky">
      <!-- directed -->
    </a>
    </td>
    <td class="tg-0pky">
      <!-- weighted -->
    </td>
    <td class="tg-0pky">
      <!-- constraints:references: penetration -->
    </a>
    </td>
    <td class="tg-0pky">
      <!-- constraints:references: angle -->
    </a>
    </td>
  </tr>
</tbody>
</table>



### Converting graphs into sequences

Graphs are converted back to timeseries by sampling node values from the graph following different strategies. Below, we provide a short snippet of code, to illustrate how this can be done.

    timegraph.to_sequence(model.ToSequenceVisitorSlidingWindow()\
    .next_node_strategy(tts.StrategySelectNextNodeRandomlyFromFirstGraph())\
    .next_value_strategy(tts.StrategyNextValueInNodeRandomForSlidingWindow().skip_every_x_steps(1))\
    .ts_length(50))\
    .draw_sequence()

When choosing the next node, the following strategies have been implemented so far: random node, random node neighbour, random node degree, random walk, random walk with restart, random walk with jump.

<>## Citing our work
<>When using this work for research purposes, we would appreciate it if the following references could be included:

