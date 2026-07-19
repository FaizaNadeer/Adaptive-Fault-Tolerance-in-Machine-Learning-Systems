# Define an adapter for the Orchestrator
class OrchestratorAdapter:
    """
    Adapter for v7 SelfHealingOrchestrator.
    """

    # Initialize the adapter with the orchestrator instance
    def __init__(self, orchestrator):
        # Store the orchestrator reference
        self.orchestrator = orchestrator

    # Define a method to trigger retraining
    def retrain(self, **kwargs):
        # Pass the retraining request down to the orchestrator
        self.orchestrator.retrain(**kwargs)
