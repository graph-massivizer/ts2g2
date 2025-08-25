from abc import ABC, abstractmethod
import numpy as np

class GraphEmbeddingStrategy(ABC):
    """Abstract base class for graph→embedding strategies. Implement embed(graph) -> np.ndarray."""

    @abstractmethod
    def embed(self, graph) -> np.ndarray:
        raise NotImplementedError
