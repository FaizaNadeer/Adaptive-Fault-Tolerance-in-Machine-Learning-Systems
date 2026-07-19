# Import the base DetectorPlugin
from self_healing_plugin.detectors.base import DetectorPlugin
# Import numpy for math operations
import numpy as np

# Define SimpleMeanShiftDetector to find basic mean shift drift
class SimpleMeanShiftDetector(DetectorPlugin):
    # Initialize with a numerical shift threshold
    def __init__(self, threshold=0.5):
        # Store the minimum acceptable mean shift
        self.threshold = threshold
        # Initialize internal state for the baseline mean
        self.baseline_mean = None

    # Define the detection logic
    def detect(self, context):
        # Calculate the mathematical mean of the incoming data batch
        current_mean = np.mean(context.X)

        # Check if this is the first execution (no baseline recorded yet)
        if self.baseline_mean is None:
            # Set the baseline mean to the current initial mean
            self.baseline_mean = current_mean
            # Return safely as no relative drift can be calculated yet
            return {"drift": False}

        # Calculate if the absolute difference exceeds the defined threshold
        drift = abs(current_mean - self.baseline_mean) > self.threshold

        # Return the structured dictionary of detected signals
        return {
            # Map the drift boolean
            "drift": drift,
            # Map the exact shift magnitude as 'uncertainty'
            "uncertainty": abs(current_mean - self.baseline_mean)
        }
