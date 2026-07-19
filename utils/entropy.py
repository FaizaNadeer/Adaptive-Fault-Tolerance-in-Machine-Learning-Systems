# Import numpy for vectorized mathematical operations
import numpy as np

# Define a calculation for Shannon entropy of predictions
def prediction_entropy(probs):
    """
    probs: array of shape (n_samples, n_classes)
    """
    # Clip absolute zeros to a tiny epsilon to prevent log(0) NaN crashes
    probs = np.clip(probs, 1e-9, 1.0)
    # Calculate - sum(p * log(p)) measuring the uncertainty in probability distribution
    return -np.mean(np.sum(probs * np.log(probs), axis=1))
