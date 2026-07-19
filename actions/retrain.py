# Import the base HealingAction class
from self_healing_plugin.actions.base import HealingAction
# Import LogisticRegression for retraining the model
from sklearn.linear_model import LogisticRegression

# Define RetrainAction inheriting from HealingAction
class RetrainAction(HealingAction):
    # Initialize with an adapter to update the underlying model
    def __init__(self, adapter):
        # Store the adapter reference
        self.adapter = adapter

    # Execute the retrain action
    def execute(self, context):
        # Import numpy for array concatenation
        import numpy as np

        # Initialize a new LogisticRegression model with balanced class weights
        model = LogisticRegression(max_iter=500, class_weight='balanced')
        
        # Online Learning: Append the new image feature mapping and its predicted label
        # to the baseline dataset, so the model legitimately learns it!
        
        # Combine historical feature data with new incoming data
        X_combined = np.vstack([context.metadata["X_train"], context.X])
        # Combine historical label data with new predicted label
        y_combined = np.concatenate([context.metadata["y_train"], context.y])

        # Fit the new model on the combined dataset
        model.fit(X_combined, y_combined)
        # Use the adapter to update the system with the newly trained model
        self.adapter.update_model(model)
        # Return success status
        return "SUCCESS"
