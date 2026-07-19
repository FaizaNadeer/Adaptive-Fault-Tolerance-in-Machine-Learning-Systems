# Define a container class for standardized error constants
class ErrorType:
    # Represents a small shift in data distribution
    SLIGHT_DRIFT = "SLIGHT_DRIFT"
    # Represents a major sudden shift or invalid sensor data
    MODERATE_FAULT = "MODERATE_FAULT"
    # Represents when the underlying concept we are trying to predict has shifted
    CONCEPT_DRIFT = "CONCEPT_DRIFT"
    # Represents when the ground truth labels are contradictory or incorrect
    LABEL_NOISE = "LABEL_NOISE"
    # Represents when a particular input feature goes missing entirely
    FEATURE_FAILURE = "FEATURE_FAILURE"
    # Represents when the model is 100% predicting a class that it shouldn't be so sure about
    OVERCONFIDENCE = "OVERCONFIDENCE"
    # Represents when an image passed in is completely unidentifiable
    UNKNOWN_IMAGE = "UNKNOWN_IMAGE"
    # Represents a clean pass with no identified issues
    NO_ERROR = "NO_ERROR"
