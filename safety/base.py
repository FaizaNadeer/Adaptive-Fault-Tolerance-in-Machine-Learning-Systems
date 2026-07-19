# Import ABC and abstractmethod for defining interface frameworks
from abc import ABC, abstractmethod

# Define a primary interface class for Data/Model safety guards
class SafetyGuard(ABC):
    """
    Prevents unsafe healing actions.
    """

    # Ensure implementation of the base allow method across subclasses
    @abstractmethod
    # Define method signature returning a strict Boolean
    def allow(self, context) -> bool:
        # Placeholder required due to abstract methodology
        pass
