# Import the base DetectorPlugin class
from self_healing_plugin.detectors.base import DetectorPlugin
# Import prediction entropy calculation function
from self_healing_plugin.utils.entropy import prediction_entropy

# Define OverconfidenceDetector to catch blindly confident predictions
class OverconfidenceDetector(DetectorPlugin):
    # Initialize with an entropy threshold
    def __init__(self, entropy_threshold=0.2):
        # Store the threshold defining when a model is "too sure"
        self.entropy_threshold = entropy_threshold

    # Define the detection logic using the context
    def detect(self, context):
        # Extract the active model from context
        model = context.model

        # Ensure the model supports probability estimation
        if not hasattr(model, "predict_proba"):
            # Return empty if model cannot yield uncertainties
            return {}

        # Predict the class probabilities for all incoming points
        probs = model.predict_proba(context.X)
        # Calculate the mathematical entropy of the probability distribution
        entropy = prediction_entropy(probs)

        # Return a dictionary wrapping the detected signals
        return {
            # Provide the exact calculated entropy scalar
            "entropy": entropy,
            # Flag bool if the entropy falls below the strict threshold
            "overconfident": entropy < self.entropy_threshold
        }
