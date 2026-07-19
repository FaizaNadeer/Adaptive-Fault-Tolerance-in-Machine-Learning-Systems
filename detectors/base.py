# Import ABC and abstractmethod for building interfaces
from abc import ABC, abstractmethod

# Define the base class for all detector plugins
class DetectorPlugin(ABC):
    """
    Detects abnormal behavior and produces signals.
    """

    # Mark the method as abstract so subclasses must implement
    @abstractmethod
    # Define detect method which takes context and returns dictionary
    def detect(self, context) -> dict:
        """
        Returns a dictionary of signals.
        Example:
        {
            "drift": True,
            "uncertainty": 0.82
        }
        """
        # Pass placeholder for abstract method
        pass
