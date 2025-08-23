from abc import ABC, abstractmethod

class PreprocessStrategy(ABC):
    """Abstract base class for preprocessing strategies. Implement transform(ts) -> Timeseries."""
    @abstractmethod
    def transform(self, ts):
        raise NotImplementedError
    