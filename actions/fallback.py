# Import HealingAction base class
from self_healing_plugin.actions.base import HealingAction
# Import NearestNeighbors from sklearn for similarity retrieval
from sklearn.neighbors import NearestNeighbors
# Import Counter to perform majority voting
from collections import Counter
# Import numpy for array manipulation
import numpy as np

# Define FallbackAction inheriting from HealingAction
class FallbackAction(HealingAction):
    """
    A safety fallback mechanism.
    If the current model prediction is highly uncertain or erratic,
    we bypass the model entirely. We take the incoming raw data (X)
    and find the K-nearest neighbors in our historically safe baseline data.
    We return the majority vote of those historical neighbors as the 'safe' prediction.
    """
    # Initialize the class with a default of 5 neighbors
    def __init__(self, n_neighbors=5):
        # Store the requested number of neighbors
        self.n_neighbors = n_neighbors

    # Execute the fallback logic
    def execute(self, context):
        # Try block to catch any mathematical or data errors
        try:
            # Extract historical baseline features from metadata
            X_train = context.metadata.get("X_train")
            # Extract historical baseline labels from metadata
            y_train = context.metadata.get("y_train")

            # Check if historical data is missing
            if X_train is None or y_train is None:
                # Return failure message if no baseline data
                return "FALLBACK_FAILED: No historical baseline available."

            # Ensure K isn't larger than our available dataset length
            k = min(self.n_neighbors, len(X_train))
            
            # Initialize a NearestNeighbors model with cosine similarity metric
            nn = NearestNeighbors(n_neighbors=k, metric='cosine')
            # Fit the model on the historical baseline data
            nn.fit(X_train)
            
            # Evaluate the first item in the incoming batch against the fitted neighbors
            distances, indices = nn.kneighbors([context.X[0]])
            
            # Fetch the actual labels of those top K historical neighbors
            neighbor_labels = y_train[indices[0]]
            
            # Find the most common label among the neighbors (majority vote)
            most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
            
            # Log the safely retrieved prediction into context signals
            context.signals["fallback_prediction"] = int(most_common_label)
            # Log the distances for observability into context signals
            context.signals["fallback_distances"] = distances[0].tolist()

            # Return success message containing the fallback prediction
            return f"FALLBACK_TRIGGERED (Predicted Class: {most_common_label})"
            
        # Catch any exceptions raised during the fallback process
        except Exception as e:
            # Return an error message detailing the failure
            return f"FALLBACK_ERROR: {str(e)}"
