__all__ = ["T2G2Pipeline"]

import numpy as np
import hashlib
from strategies.g2embedding import get_graph_embedding
from core.time_series import Timeseries 
from strategies.preprocess import get_preprocess_strategy
from core.time_series import Timeseries
from core.graphs import TimeGraph
from strategies.view import get_view_strategy
from strategies.merge import get_combine_indentical_strategy, LinkGraphs
import networkx as nx
import pandas as pd


class T2G2Pipeline:
    def __init__(self, attribute = 'value'):
        self.ts = None
        self.graphs = [] # list of dicts per dimension; values are TimeGraph (window-level subgraphs)
        self.Graph = None # fused node-level nx.Graph after linking; may be None until link
        self.meta_G = None  # meta-graph of windows; nodes are window subgraphs; used by g2ts
        self.embedding = None
        self.ts_views = None
        self.graph_order = []
        self.sequences = None
        self.timegraph = None
        
      
        self.attribute = attribute
        self.histogram_frequencies = None
        self.histogram_bins = None
        self.w = 1
        self.tau = 1
        self.is_implemented = True
        self.quantiles = None
        self.quantile_values = None
        

    def load_timeseries(self, ts_data):
        """
        Load time series data into the pipeline. Accepts either a Timeseries object,
        a 1D array/list for single-dimension, a 2D array for multi-dimension (each row one dim),
        or a list of 1D arrays/lists/Timeseries for multi-dimension. 

        The original single Timeseries (if applicable) is stored in `ts`; subsequent steps use `ts_views`
        as the evolving container with nested views: dims -> windows -> Timeseries.

        Parameters:
            ts_data (Timeseries | np.ndarray): Input time series data.

        Returns:
            self: Enables method chaining.
        """
        # Case 1: already a Timeseries (single dimension)
        if isinstance(ts_data, Timeseries):
            self.ts = ts_data
            self.ts_views = [[self.ts]]
        # Case 2: numpy arrays
        elif isinstance(ts_data, np.ndarray):
            if ts_data.ndim == 1:
                self.ts = Timeseries(data=ts_data)
                self.ts_views = [[self.ts]]
            elif ts_data.ndim == 2:
                # Interpret as multi-dimensional: each row is one dimension
                dims = []
                for i in range(ts_data.shape[0]):
                    dims.append([Timeseries(data=ts_data[i])])
                self.ts = None
                self.ts_views = dims
            else:
                raise ValueError("Only 1D or 2D numpy arrays are supported for load_timeseries().")
        elif isinstance(ts_data, list):
            raise TypeError("List input is not supported in load_timeseries(). Please convert your data to a numpy array: 1D for single-dimension, 2D (each row one dimension) for multi-dimension.")
        else:
            raise TypeError(f"Unsupported type for time series: {type(ts_data)}")

        # Reset/initialize downstream state
        self.meta_G = None
        return self

    def preprocess(self, strategy=None, **kwargs):

        if strategy is None:
            # No-op: ensure nested structure exists
            if self.ts_views is None:
                self.ts_views = [[self.ts]]
            return self
        print(f"Preprocessing with method: {strategy}")
        if not hasattr(strategy, 'transform'):
            raise TypeError("preprocess strategy must implement a .transform(Timeseries, **kwargs) method")

        # Ensure we have a nested structure to iterate
        current_views = self.ts_views or [[self.ts]]
        transformed_nested = []
        for dim_views in current_views:
            transformed_dim = []
            for ts in dim_views:
                if not isinstance(ts, Timeseries):
                    raise TypeError("preprocess expects Timeseries elements inside ts_views.")
                transformed_ts = strategy.transform(ts, **kwargs)
                if not isinstance(transformed_ts, Timeseries):
                    raise TypeError("preprocess strategy must return a Timeseries instance.")
                transformed_dim.append(transformed_ts)
            transformed_nested.append(transformed_dim)

        # Update state: replace views
        self.ts_views = transformed_nested
        return self

    def viewing(self, strategy=None):
        """Split current time series into views/windows via a strategy.

        This method is **chainable**: it can be called multiple times. On each call,
        the strategy is applied to **every window** in each current dimension, and
        outputs are aggregated per dimension while preserving order.

        Accepted strategy return types per input Timeseries:
            - List[Timeseries] (single-dimension expansion), or
            - List[List[Timeseries]] (multi-dimension expansion from one input).

        The internal storage is always nested: dims -> windows -> Timeseries.
        """
       
        if strategy in (None, 'none'):
            print("No view strategy specified; using current ts_views as-is.")
            if self.ts_views is None:
                self.ts_views = [[self.ts]]
            return self

        print(f"Applying viewing strategy: {strategy}")
        if not hasattr(strategy, 'apply'):
            raise TypeError("viewing strategy must implement an .apply(Timeseries, **kwargs) method")

        # Ensure we have a nested structure to iterate
        current_views = self.ts_views or [[self.ts]]
        next_nested = []  # will become the new dims->windows

        for dim_idx, dim_views in enumerate(current_views):
            # aggregated results for this dimension
            # could be single-dim (index 0) or multi-dim (multiple lists)
            aggregated_dims = []  # List[List[Timeseries]]

            for win_idx, win_ts in enumerate(dim_views):
                if not isinstance(win_ts, Timeseries):
                    raise TypeError("viewing expects Timeseries elements inside ts_views.")
                produced = strategy.apply(win_ts)
                if produced is None:
                    raise ValueError("Viewing strategy returned None; expected a list of Timeseries or list of lists.")

                # Normalize "produced" into a List[List[Timeseries]] shape for aggregation
                if isinstance(produced, (list, tuple)) and len(produced) > 0 and isinstance(produced[0], Timeseries):
                    produced_dims = [list(produced)]  # single-dimension expansion
                elif isinstance(produced, (list, tuple)) and len(produced) > 0 and isinstance(produced[0], (list, tuple)):
                    produced_dims = [list(x) for x in produced]  # multi-dimension expansion
                else:
                    raise TypeError("Viewing strategy must return List[Timeseries] or List[List[Timeseries]].")

                # Aggregate per dimension; if the first result defines N dims, subsequent windows
                # must also return N dims for this original dimension.
                if not aggregated_dims:
                    aggregated_dims = [list(x) for x in produced_dims]
                else:
                    if len(aggregated_dims) != len(produced_dims):
                        raise ValueError(
                            f"Inconsistent number of dimensions returned by viewing strategy within the same original dimension: "
                            f"got {len(produced_dims)} but expected {len(aggregated_dims)}.")
                    for k in range(len(produced_dims)):
                        aggregated_dims[k].extend(produced_dims[k])

            # After aggregating all windows within this original dimension, append to next_nested
            next_nested.extend(aggregated_dims)

        self.ts_views = next_nested
        print(f"Number of dimensions: {len(self.ts_views)}; total windows: {sum(len(x) for x in self.ts_views)}")
        return self

    def add_view(self, strategy):
        """Add an additional dimension by applying a view strategy on the base time series.

        This mirrors the legacy `.add(...)` behavior by **horizontally expanding** the
        `ts_views` structure with a new dimension produced from the original
        single-dimension time series `self.ts`.

        Parameters
        ----------
        strategy : object
            Must implement `apply(ts: Timeseries)` and return either a
            `List[Timeseries]` (single-dimension expansion) or a
            `List[List[Timeseries]]` (multi-dimension expansion).
        """
        # We intentionally require a base single-dimension Timeseries for clarity.
        # If the pipeline was initialized with a multi-dimensional array, `self.ts`
        # will be None; in that case, ask the caller to provide an explicit base.
        if self.ts is None:
            raise ValueError(
                "add_view(...) requires a base single-dimension Timeseries loaded via "
                "load_timeseries(...). For multi-dimensional inputs, please load your "
                "extra dimension explicitly or refactor accordingly."
            )

        if not hasattr(strategy, 'apply'):
            raise TypeError("add_view strategy must implement an .apply(Timeseries) method")

        # Ensure nested structure is initialized
        if self.ts_views is None:
            self.ts_views = [[self.ts]]

        produced = strategy.apply(self.ts)
        if produced is None:
            raise ValueError(
                "add_view strategy returned None; expected a list of Timeseries or list of lists."
            )

        # Normalize the produced output into List[List[Timeseries]]
        if isinstance(produced, (list, tuple)) and len(produced) > 0 and isinstance(produced[0], Timeseries):
            new_dim = [list(produced)]  # single-dimension expansion
        elif isinstance(produced, (list, tuple)) and len(produced) > 0 and isinstance(produced[0], (list, tuple)):
            new_dim = [list(x) for x in produced]  # multi-dimension expansion
        else:
            raise TypeError(
                "add_view expects the strategy to return List[Timeseries] or List[List[Timeseries]]."
            )

        # Append as a new dimension
        self.ts_views.extend(new_dim)
        return self


    def _hash(self, graph):
        """Returns unique hash of this graph."""
        str_to_hash = str(graph.nodes()) + str(graph.edges())
        return hashlib.md5(str_to_hash.encode()).hexdigest()
    
    def to_graph(self, strategy):
        pool = ["#000000", "#101010", "#202020", "#303030", "#404040", "#505050", "#606060", "#707070", "#808080", "#909090", "#a0a0a0", "#b0b0b0", "#c0c0c0", "#d0d0d0", "#e0e0e0", "#f0f0f0"]
        import math
        num = math.floor(16/len(self.ts_views))
        if len(self.ts_views) > 16:
            num = 1
        color_counter = 0
        self.w, self.tau = strategy._get_w_tau()
        for ts in self.ts_views:
            graph_dict = {}
            order=[]

            counter = 0
            for timeseries in ts:
                g =  strategy.to_graph(timeseries)
                g = g.graph
                
                if strategy._has_value():
                    for i in range(len(g.nodes)):
                        old_value = g.nodes[i][self.attribute]
                        new_value = [old_value]
                        g.nodes[i][self.attribute] = new_value
                hash = self._hash(g)
                mapping = {node: f"{hash}_{node}" for node in g.nodes}
                g = nx.relabel_nodes(g, mapping)

                nx.set_node_attributes(g, pool[color_counter], "color")
                color_counter += num
                if color_counter > 15:
                    color_counter = 0
                

                nx.set_edge_attributes(g, strategy._get_name(), "strategy")
                graph_dict[self._hash(g) + f"_{counter}"] = g
                order.append(self._hash(g) + f"_{counter}")

                if self.Graph == None:
                    self.Graph = g

                counter+=1

            self.graphs.append(graph_dict)
            self.graph_order.append(order)
        
        self.quantiles, self.quantile_values = strategy._get_bins()
        
        if (len(self.ts_views) == 1 and len(self.ts_views[0]) == 1):
            self.timegraph = TimeGraph(self.Graph, graphs = self.graphs[0], is_implemented=strategy._has_implemented_to_ts(), histogram_frequencies = self.histogram_frequencies, histogram_bins = self.histogram_bins, w = self.w, tau = self.tau, quantiles = self.quantiles, quantile_values=self.quantile_values)
            return self
        else:
            self.is_implemented = strategy._has_implemented_to_ts()
            return self
                           
    def link_graphs(self, strategy):
        """Fuse multiple window-based graphs into a single unified graph."""
        self.timegraph =  TimeGraph(strategy.link(self.graphs, self.graph_order, self.ts_views), graphs = self.graphs, is_implemented=self.is_implemented, histogram_frequencies = self.histogram_frequencies, histogram_bins = self.histogram_bins, w = self.w, tau = self.tau, quantiles = self.quantiles, quantile_values=self.quantile_values)
        return self
        

    def link_nodes_within_graphs(self, strategy):
        """Apply in-graph node-linking strategies to each subgraph.

        Args:
            strategy: a pre-configured LinkNodesWithinGraph instance that will be applied directly.
        Returns:
            self
        """
        if self.timegraph is None:
            raise ValueError("No TimeGraph available. Run to_graph(...) first.")
        self.timegraph.link(strategy)
        return self

    def combine_identical_subgraphs(self):
        if self.timegraph is None:
            raise ValueError("No TimeGraph available. Run to_graph(...) first.")
        self.timegraph.combine_identical_subgraphs()
        return self
        
    
    def add_edge(self, node_1, node_2, weight=None):
        """Adds edge between node_1 and node_2."""
        if self.timegraph is None:
            raise ValueError("No TimeGraph available. Run to_graph(...) first.")
        self.timegraph.add_edge(node_1, node_2, weight=None)
        return self
        

    def combine_identical_nodes(self):
        if self.timegraph is None:
            raise ValueError("No TimeGraph available. Run to_graph(...) first.")
        self.timegraph.combine_identical_nodes()
        return self

    def to_embedding(self, strategy):
        """Convert the (unified) graph to an embedding using a specified method."""
        if self.timegraph is None:
            raise ValueError("No TimeGraph available. Run to_graph(...) first.")
        self.timegraph.to_embedding(strategy)
        return self
    
    def to_ts_embedding(self, encoder):
        #for future extensions, e.g., if we want to use a different encoder
        self.ts_embedding = encoder.encode(self.ts)   # or encode(self.ts_views) 
        return self
    
    def to_graph_embedding(self, encoder):
        #for future extensions, e.g., if we want to use a different encoder
        if self.Graph is None:
            raise RuntimeError("Graph not built; call to_graph/merge/link first.")
        self.graph_embedding = encoder.encode(self.Graph)
        return self

    def get_embedding(self):
        return self.timegraph.get_embedding()
    
    def to_histogram(self, bins):
        self.histogram_frequencies = []
        self.histogram_bins = []
        for i in range(len(self.ts)):
            a, b = np.histogram(self.ts[i], bins = bins)
            self.histogram_frequencies.append(a)
            self.histogram_bins.append(b)

        return self
    
    def to_timeseries(self, strategy):
        """Reconstruct/emit time series from the current graph(s) using a strategy.

        For g2ts strategies (esp. SlidingWindow variants), we must pass a meta-graph
        whose *nodes are the window subgraphs themselves*, not the fused node-level graph.
        """
        if self.timegraph is None:
            raise ValueError("No TimeGraph available. Run to_graph(...) first.")
        self.timegraph = self.timegraph.to_sequence(strategy)
        self.sequences = self.timegraph.sequences
        return self
    
            

    def get_timeseries(self):
        """Return the originally loaded Timeseries (or None if multi-dimensional input)."""
        return self.ts
    
    
    def get_Graph(self):
        """
        Retrieve the generated graph object.

        Returns:
            TimeGraph | None: The generated graph, or None if not yet created.
        """
        return self.timegraph.get_graph()
            
       
if __name__ == "__main__":
    pass