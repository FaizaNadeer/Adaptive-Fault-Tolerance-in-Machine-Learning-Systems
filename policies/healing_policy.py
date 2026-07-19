# Import standard set of enumerated ErrorTypes
from self_healing_plugin.errors.error_types import ErrorType
# Import the handler for executing retraining actions
from self_healing_plugin.actions.retrain import RetrainAction
# Import the handler for blocking dangerous actions
from self_healing_plugin.actions.block import BlockAction

# Define a baseline hardcoded healing policy
class DefaultHealingPolicy:
    """
    A simple, hardcoded policy engine.
    This serves as a default implementation or fallback if no configuration is provided.
    It maps error types directly to actions without dynamic configuration.
    """
    
    # Initialize the default concrete policy 
    def __init__(self, adapter):
        # Store the adapter to pass to actions that need it (e.g., RetrainAction).
        self.adapter = adapter

    # Define the core logic mapping signals to operations
    def decide(self, error_type, signals=None):
        """
        Decide the action to take based on the error type.
        
        Args:
            error_type (str): The detected error type (e.g., DATA_DRIFT).
            signals (dict, optional): Additional context (unused in this simple policy).
            
        Returns:
            HealingAction: The action to execute.
        """
        
        # If Data Drift is detected, attempt to continuously learn safely
        if error_type in (ErrorType.SLIGHT_DRIFT, ErrorType.MODERATE_FAULT):
            # Re-train using the provided model adapter
            return RetrainAction(self.adapter)

        # For excessive chaos or untrustworthy confidence scores
        if error_type in (
            ErrorType.OVERCONFIDENCE,
            ErrorType.LABEL_NOISE,
        ):
            # Block the operation and prevent the engine from learning garbage
            return BlockAction()

        # For any other unknown error types, fail-safe blocking
        return BlockAction()
