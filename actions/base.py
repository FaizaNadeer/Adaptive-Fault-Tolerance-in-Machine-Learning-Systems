# Import ABC and abstractmethod for abstract base classes
from abc import ABC, abstractmethod

# Define an abstract base class for healing actions
class HealingAction(ABC):
    """
    Executes a healing step.
    """

    # Mark the execute method as an abstract method
    @abstractmethod
    # The execute method must be implemented by subclasses, accepting a context
    def execute(self, context):
        """
        Must return: SUCCESS | FAIL | BLOCKED
        """
        # Pass placeholder implementation for abstract method
        pass
