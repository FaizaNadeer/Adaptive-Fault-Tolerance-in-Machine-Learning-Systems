# self_healing_plugin/policies/configurable_policy.py

import json
from self_healing_plugin.policies.actions import PolicyAction
from self_healing_plugin.actions.retrain import RetrainAction
from self_healing_plugin.actions.block import BlockAction

class ConfigurableHealingPolicy:
    """
    A policy engine that determines the appropriate healing action based on a 
    loaded configuration file. This allows developers to define behavior declaratively.
    """

    def __init__(self, adapter, config_path):
        # Initialize the policy engine with an adapter (for actions like retraining)
        # and a path to the configuration file.
        
        # Load the policy configuration from the JSON file.
        # This map defines how to handle each ErrorType.
        with open(config_path, "r") as f:
            self.policy_map = json.load(f)
            
        # Store the adapter instance, which provides the actual implementation
        # for actions like updating the model.
        self.adapter = adapter

    def decide(self, error_type, signals=None):
        """
        Decide which action to take based on the error type and optional signals.
        
        Args:
            error_type (str): The classified error type (e.g., "DATA_DRIFT").
            signals (dict, optional): Additional metadata about the error (e.g., confidence).
        
        Returns:
            HealingAction: An instantiated action object (e.g., RetrainAction, BlockAction).
        """
        
        # Ensure signals is a dictionary if not provided, for safe key access.
        if signals is None:
            signals = {}

        # Look up the policy rule for the given error type in the loaded map.
        rule = self.policy_map.get(error_type)

        # If no rule is defined for this error type, default to safety.
        # Blocking is the safest default behavior when behavior is undefined.
        if not rule:
            return BlockAction()

        # Extract the declared action string from the rule (e.g., "RETRAIN").
        action = rule.get("action")

        # --- Safety/Condition Checks ---
        
        # Check if a minimum confidence threshold is required for this action.
        min_conf = rule.get("min_confidence")
        if min_conf is not None:
            # Calculate confidence from entropy. 
            # High entropy implies high uncertainty (low confidence).
            # We assume signals contains 'entropy'. If missing, default to 1.0 (max uncertainty).
            entropy = signals.get("entropy", 1.0) 
            
            # Simple conversion: Confidence is the inverse of entropy.
            confidence = 1.0 - entropy
            
            # If confidence is below the required threshold, we abort the risky action.
            if confidence < min_conf:
                # Return BlockAction to stop processing safely.
                return BlockAction()

        # --- Action Instantiation ---

        # If the policy dictates RETRAIN, return a RetrainAction.
        # Pass the adapter so the action can interact with the model/orchestrator.
        if action == PolicyAction.RETRAIN:
            return RetrainAction(self.adapter)

        # If the policy dictates NO_ACTION (or explicit BLOCK), return BlockAction.
        # Currently, NO_ACTION maps to BlockAction to ensure no side effects occur.
        if action == PolicyAction.NO_ACTION:
            return BlockAction()

        # Default fallback for any unhandled action strings: Block.
        return BlockAction()
