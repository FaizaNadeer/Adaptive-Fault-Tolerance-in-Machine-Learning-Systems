# Import the abstract base class HealingAction
from self_healing_plugin.actions.base import HealingAction

# Define a class BlockAction that inherits from HealingAction
class BlockAction(HealingAction):
    """
    Represents a safe no-op action when healing is blocked.
    """

    # Implement the execute method for BlockAction
    def execute(self, context):
        # Set the action metadata property to 'blocked'
        context.metadata["action"] = "blocked"
        # Return the string 'BLOCKED' as the execution result
        return "BLOCKED"
