from abc import ABC, abstractmethod
from core import Timeseries

class BaseViewStrategy(ABC):
    @abstractmethod
    def apply(self, time_series: Timeseries):
        raise NotImplementedError("Must be implemented by subclass.")
