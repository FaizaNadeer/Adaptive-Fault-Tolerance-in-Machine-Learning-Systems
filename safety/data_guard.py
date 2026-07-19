# Define a simple guard validating whether data can even be used
class DataValidityGuard:
    # Function accepting the complete execution context
    def allow(self, context):
        # Safely attempt to extract the ground-truth baseline labels
        y = context.metadata.get("y_train")
        # Ensure 'y' actually exists and contains at least 2 unique classes for comparison
        return y is not None and len(set(y)) >= 2
