# Import the base DetectorPlugin
from self_healing_plugin.detectors.base import DetectorPlugin

# Define GPRDriftDetector inheriting from DetectorPlugin
class GPRDriftDetector(DetectorPlugin):
    """
    Wraps existing GPR uncertainty-based drift detection.
    """

    # Initialize with a GPR model and a threshold
    def __init__(self, gpr_model, uncertainty_threshold=0.7):
        # Store the provided GPR model instance
        self.gpr = gpr_model
        # Store the uncertainty threshold
        self.threshold = uncertainty_threshold

    # Define the detect method logic
    def detect(self, context):
        # Call the existing detect method on the underlying GPR model
        drift, uncertainty = self.gpr.detect(context.X)

        # Return the structured dictionary of detected signals
        return {
            # Map the drift boolean
            "drift": drift,
            # Map the uncertainty scalar
            "uncertainty": uncertainty
        }
