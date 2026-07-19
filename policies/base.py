# Import the necessary abstract base classes framework
from abc import ABC, abstractmethod

# Define the base interface for the Policy Engine
class PolicyEngine(ABC):
    """
    Maps error types to healing actions.
    """

    # Enforce implementation of the decide method in all subclasses
    @abstractmethod
    # Define method signature to accept an error_type string
    def decide(self, error_type: str):
        # Empty placeholder required by abstract methodology
        pass
