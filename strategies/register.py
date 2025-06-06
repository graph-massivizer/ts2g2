from strategies.exceptions import StrategyNotImplementedError
from collections import defaultdict

# Global strategy registry
STRATEGY_REGISTRY = defaultdict(dict)

def register_strategy(category, name):
    """
    Decorator to register a strategy function under a given category and name.

    Example:
        @register_strategy("ts2g", "visibility")
        def visibility(...):
            ...
    """
    def decorator(func):
        if name in STRATEGY_REGISTRY[category]:
            raise ValueError(f"Strategy '{name}' already registered under category '{category}'")
        STRATEGY_REGISTRY[category][name] = func
        return func
    return decorator

def get_strategy(category, name):
    """
    Retrieve a registered strategy function by category and name.
    """
    if category not in STRATEGY_REGISTRY or name not in STRATEGY_REGISTRY[category]:
        raise StrategyNotImplementedError(category, name)
    return STRATEGY_REGISTRY[category][name]

def list_strategies(category=None):
    """
    List all registered strategies, or strategies for a specific category.
    """
    if category:
        return list(STRATEGY_REGISTRY[category].keys())
    return {cat: list(strats.keys()) for cat, strats in STRATEGY_REGISTRY.items()}