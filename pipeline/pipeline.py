__all__ = ["T2G2Pipeline"]

import numpy as np
from strategies.g2embedding import get_graph_embedding
from core.time_series import Timeseries 
from strategies.preprocess import get_preprocess_strategy
from core.time_series import Timeseries
from strategies.view import get_view_strategy
from strategies.merge import get_merge_strategy


class T2G2Pipeline:
    def __init__(self, config=None):
        self.config = config or {}
        self.ts = None
        self.graphs = None #subgraphs
        self.Graph = None # merged big graph
        self.embedding = None
        self.ts_views = None

    def load_timeseries(self, ts_data):
        """
        Load time series data into the pipeline. Accepts either a Timeseries object
        or a raw array/list, which will be automatically wrapped into a Timeseries instance.

        Parameters:
            ts_data (Timeseries | np.ndarray | list): Input time series data.

        Returns:
            self: Enables method chaining.
        """
        if isinstance(ts_data, Timeseries):
            self.ts = ts_data
        elif isinstance(ts_data, (np.ndarray, list)):
            self.ts = Timeseries(data=ts_data)
        else:
            raise TypeError(f"Unsupported type for time series: {type(ts_data)}")
        return self

    def preprocess(self, method="none", **kwargs):
        if method is None or method == "none":
            print("Skipping preprocessing.")
            return self
        print(f"Preprocessing with method: {method}")
        preprocess_func = get_preprocess_strategy(method)
        self.ts = preprocess_func(self.ts)
        return self
    
    def apply_view_strategy(self, view_strategy="none", **kwargs):
        """
        Apply view/windowing strategy to the current Timeseries.

        Parameters:
            view_strategy (str): The registered viewing strategy name.
            kwargs: Additional parameters passed to the viewing strategy.

        Returns:
            list[Timeseries]: A list of time series segments (windows).
        """
        if not isinstance(self.ts, Timeseries):
            raise TypeError("apply_view_strategy only supports static Timeseries at this stage.")

        if view_strategy == "none":
            print("No view strategy specified; applying TS2G transformation to full time series.")
            self.ts_views = [self.ts]
            return self

        print(f"Applying viewing strategy: {view_strategy}")
        view_fn = get_view_strategy(view_strategy, **kwargs)
        self.ts_views = view_fn.apply(self.ts)
        print(f"Length of veiw is :{len(self.ts_views)}")
        return self
    
    def to_graph(self, ts2g_strategy):
        print(f"Transforming time series to graph with strategy: {ts2g_strategy}")
        self.graphs = [ts2g_strategy.to_graph(view) for view in self.ts_views] 
        print(f"Number of graphs is: {len(self.graphs)}")

        return self

    

    def merge(self, method="combine_identical_subgraphs", **kwargs):
        """
        Fuse multiple window-based graphs into a single unified graph.

        Parameters:
            method (str): Fusion strategy name registered under 'merge'.
            kwargs: Additional parameters passed to the fusion strategy.

        Returns:
            self: Enables method chaining.
        """
        if not isinstance(self.graphs, list):
            print("graphs should be a list.")
            return self
        if len(self.graphs) == 1:
            print("No need to fuse; graph is already a single object.")
            self.Graph = self.graphs[0]
            return self

        print(f"Fusing {len(self.graphs)} graphs using strategy: {method}")
        fuse_func = get_merge_strategy(method, **kwargs)
        self.Graph = fuse_func(self.Graph, self.graphs)
        return self

    def get_Graph(self):
        """
        Retrieve the generated graph object.

        Returns:
            TimeGraph | None: The generated graph, or None if not yet created.
        """
        return self.Graph

    def to_embedding(self, method="eigenvector"):
        """
        Convert the graph to an embedding using a specified method.
        
        Parameters:
            method (str): Embedding method name. Must match a registered embedding strategy.
        
        Returns:
            self: Enables method chaining.
        """
        print(f"Encoding graph to embedding using '{method}' strategy...")
        embedding_func = get_graph_embedding(method=method)
        self.embedding = embedding_func(self.Graph)
        return self

    def run(self):
        # Return embedding if available; otherwise return graph
        if self.embedding is not None:
            return self.embedding
        return self.Graph

    def to_timeseries(self, reconstructor):
        # Assume reconstructor has a .reconstruct(graph) method
        print("Reconstructing time series from graph...")
        self.ts = reconstructor.reconstruct(self.graphs)
        return self

    def run_stream(self, ts_generator, strategy, embedding_model, preprocess_method="standard", **kwargs):
        for ts_chunk in ts_generator:
            yield (
                self.load_timeseries(ts_chunk)
                    .preprocess(method=preprocess_method, **kwargs)
                    .to_graph(strategy)
                    .to_embedding(embedding_model)
                    .run()
            )
            
       
if __name__ == "__main__":
    pass