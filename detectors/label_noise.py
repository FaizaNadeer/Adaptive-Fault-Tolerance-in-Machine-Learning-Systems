# Import numpy for numerical array operations
import numpy as np
# Import the base DetectorPlugin
from self_healing_plugin.detectors.base import DetectorPlugin

# Define LabelNoiseDetector to identify mislabeled data
class LabelNoiseDetector(DetectorPlugin):
    # Initialize with a disagreement threshold
    def __init__(self, disagreement_threshold=0.3):
        # Store the threshold determining when noise becomes problematic
        self.disagreement_threshold = disagreement_threshold

    # Define the core detection method
    def detect(self, context):
        # Check if actual labels are provided in the context
        if context.y is None:
            # Return an empty dictionary if no labels exist to compare against
            return {}

        # Extract the model from the context
        model = context.model
        # Check if the model has a prediction method
        if not hasattr(model, "predict"):
            # Return empty signals if the model is incapable of predicting
            return {}

        # Generate predictions for the incoming features
        preds = model.predict(context.X)
        # Calculate the mathematical mean of mismatched labels
        disagreement = np.mean(preds != context.y)

        # Return a dictionary of the calculated signals
        return {
            # Provide the exact scalar of label disagreement
            "label_disagreement": disagreement,
            # Provide a boolean flag if the noise exceeds the acceptable threshold
            "label_noise": disagreement > self.disagreement_threshold
        }
