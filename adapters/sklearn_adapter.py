# Define an adapter for Scikit-Learn models
class SklearnAdapter:
    # Initialize with a model holder dictionary
    def __init__(self, model_holder):
        # Store the reference to the dictionary holding the active model
        self.model_holder = model_holder

    # Define a method to update the active model
    def update_model(self, model):
        # Replace the active model in the dictionary with the newly trained one
        self.model_holder["model"] = model
