# Import the base DetectorPlugin class
from self_healing_plugin.detectors.base import DetectorPlugin
# Import numpy for array manipulation and math
import numpy as np

# Define OutOfDistributionDetector to identify unknown inputs
class OutOfDistributionDetector(DetectorPlugin):
    """
    Detects if an incoming sample is completely out of distribution
    by evaluating the model's highest confidence score.
    """
    # Initialize with a designated confidence threshold
    def __init__(self, confidence_threshold=0.65):
        # Store the minimum acceptable confidence score
        self.confidence_threshold = confidence_threshold

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
        # Find the maximum confidence score generated for each point
        max_conf = np.max(probs, axis=1)
        
        # If the highest confidence for any class is below our threshold,
        # the model is entirely unsure. It's an unknown image.
        
        # Determine if any max probability falls short of the threshold
        is_ood = bool(np.any(max_conf < self.confidence_threshold))

        # Return a dictionary wrapping the detected OOD signals
        return {
            # Provide the calculated average of the maximum confidences
            "max_confidence": float(np.mean(max_conf)),
            # Flag whether the input was classified as OOD
            "ood": is_ood
        }
