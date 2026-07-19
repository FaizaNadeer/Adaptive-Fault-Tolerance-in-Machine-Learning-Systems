# Define HealingContext to wrap all data passing through the plugin
class HealingContext:
    """
    Shared state passed across detectors, policies,
    safety guards, and actions.
    """

    # Initialize the context with standard data and optional fields
    def __init__(
        self,
        model,
        X=None,
        y=None,
        signals=None,
        metadata=None
    ):
        # Store the active machine learning model
        self.model = model
        # Store the incoming feature data point(s)
        self.X = X
        # Store the incoming label data point(s)
        self.y = y

        # Store any detected signals (defaulting to empty dictionary)
        self.signals = signals or {}
        # Store any historical dataset baseline or other metadata
        self.metadata = metadata or {}

        # Initialize the error type classification to None
        self.error_type = None
        # Initialize the policy decision to None
        self.decision = None
        # Initialize the execution result metadata to None
        self.result = None
