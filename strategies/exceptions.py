class StrategyNotImplementedError(Exception):
    """Raised when a requested strategy is not found in the registry."""
    def __init__(self, category: str, name: str):
        message = f"Strategy '{name}' in category '{category}' is not implemented or not registered."
        super().__init__(message)
        self.category = category
        self.name = name
