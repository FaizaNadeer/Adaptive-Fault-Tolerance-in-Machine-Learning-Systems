# Import the enumerated ErrorType to standardize outputs
from self_healing_plugin.errors.error_types import ErrorType

# Define a concrete classifier to map signals to literal errors
class RuleBasedErrorClassifier:
    # Define the classification logic method taking a signals dict
    def classify(self, signals):
        # Check if Out-Of-Distribution signal fired
        if signals.get("ood"):
            # Return UNKNOWN_IMAGE if the input is totally foreign
            return ErrorType.UNKNOWN_IMAGE
            
        # Check if general data drift signal fired
        if signals.get("drift"):
            # Extract the magnitude of the drift, defaulting to 0
            magnitude = signals.get("uncertainty", 0)
            
            # Use thresholding to differentiate between severity of drift
            if magnitude > 10.0:  
                # If massive shift, classify as MODERATE_FAULT
                return ErrorType.MODERATE_FAULT
            else:
                # If small shift, classify as SLIGHT_DRIFT
                return ErrorType.SLIGHT_DRIFT

        # Check if model confidence is dangerously high 
        if signals.get("overconfident"):
            # Return OVERCONFIDENCE error
            return ErrorType.OVERCONFIDENCE

        # Check if the labels seem inherently contradictory
        if signals.get("label_noise"):
            # Return LABEL_NOISE error
            return ErrorType.LABEL_NOISE

        # If no strict error signals fired, return NO_ERROR
        return ErrorType.NO_ERROR
