from typing import Protocol

from core.graphs import BaseTimeGraph
from core.time_series import Timeseries
from abc import ABC, abstractmethod

"""Minimal interfaces for the ts→graph framework.

Users implement a *strategy* (create(ts: Timeseries) -> BaseTimeGraph) and a *builder* that
returns a strategy instance via get_strategy(). Keep this file tiny to avoid
circular imports.
"""

class TimeseriesToGraphStrategy(ABC):
    """Abstract base class for strategies. Subclasses must implement create(ts: Timeseries) and return a BaseTimeGraph."""
    @abstractmethod
    def to_graph(self, ts: Timeseries) -> BaseTimeGraph:
        raise NotImplementedError

class StrategyBuilder(Protocol):
    """All builders must return a strategy instance via get_strategy().
    Optional fluent setters (e.g., with_xxx()) can be added by implementations.
    """
    def get_strategy(self) -> TimeseriesToGraphStrategy: ...

__all__ = [
    "TimeseriesToGraphStrategy",
    "StrategyBuilder",
]