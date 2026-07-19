# Import json for parsing configuration files
import json
# Import os for resolving file paths
import os
# Import the abstract base class PolicyEngine
from self_healing_plugin.policies.base import PolicyEngine
# Import the actual RetrainAction payload
from self_healing_plugin.actions.retrain import RetrainAction
# Import the actual BlockAction payload
from self_healing_plugin.actions.block import BlockAction

# Define a concrete policy engine driven completely by a JSON map
class ConfigDrivenHealingPolicy(PolicyEngine):
    # Initialize with an adapter and an optional custom config path
    def __init__(self, adapter, config_path=None):
        # Store the adapter for actions to manipulate the model
        self.adapter = adapter
        # Call private loader to extract and store the dictionary map
        self.config = self._load_config(config_path)

    # Define private method for safely loading configuration JSON
    def _load_config(self, config_path):
        # Verify if a custom path was not provided
        if not config_path:
            # Default to the one in the project root
            config_path = os.path.join(
                # Traverse up directories to find root level config
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                # Exact target filename
                "policy_config.json"
            )
        # Attempt disk read operation
        try:
            # Open up file handle in readonly mode
            with open(config_path, "r") as f:
                # Return parsed JSON objects cleanly
                return json.load(f)
        # Catch arbitrary errors, most commonly format/file not found
        except Exception as e:
            # Print warning but don't hard crash the plugin init
            print(f"Warning: Could not load policy config: {e}. Falling back to safe defaults.")
            # Return empty config map
            return {}

    # Define the core decision mapping logic
    def decide(self, error_type, signals=None):
        # Standardize the incoming error classification to a string
        error_name = str(error_type)
        # Fetch dictionary properties explicitly mapped for this error type
        policy_rule = self.config.get(error_name, {})
        # Extract the action string, defaulting safely to block
        action_name = policy_rule.get("action", "BLOCK") # Default safe action

        # If rule explicitly demands we retrain the AI
        if action_name == "RETRAIN":
            # Instantiate and pass the required adapter
            return RetrainAction(self.adapter)
        # If the rule demands falling back to similarity historic retrieval    
        elif action_name == "FALLBACK":
            # Delay import to avoid circular dependencies if any
            from self_healing_plugin.actions.fallback import FallbackAction
            # Return new fallback action handler instance
            return FallbackAction()
        # If the rule explicitly commands no action taken
        elif action_name == "NO_ACTION":
            # If no action is needed, we still "block" any self-healing overrides
            return BlockAction()
        # Default safety net for all unnamed strings
        else:
            # Return default BlockAction
            return BlockAction()
