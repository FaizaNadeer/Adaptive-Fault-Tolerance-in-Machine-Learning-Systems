# self_healing_plugin/policies/actions.py

# Define a container class for policy action constants
class PolicyAction:
    """
    Defines the standard set of allowed actions that a policy can return.
    Using constants prevents typos and ensures type safety across the policy engine.
    """
    
    # Action to retrain the model on the new data
    RETRAIN = "RETRAIN"
    
    # Action to block the current inference or process flow
    BLOCK = "BLOCK"
    
    # Action to take no specific intervention (implies proceeding as normal or doing nothing)
    NO_ACTION = "NO_ACTION"
