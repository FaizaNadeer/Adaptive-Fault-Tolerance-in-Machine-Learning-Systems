# Import numpy for high performance matrix calculations
import numpy as np
# Import sophisticated statistical functions from scipy
from scipy.stats import entropy, wasserstein_distance

# Define a function to calculate KL Divergence metric
def calculate_kl_divergence(p, q):
    """
    Calculates the Kullback-Leibler divergence between two distributions.
    Useful for comparing probability distributions (e.g. classification outputs).
    """
    # Convert input list P to a numpy array for float ops
    p = np.asarray(p, dtype=float)
    # Convert input list Q to a numpy array for float ops
    q = np.asarray(q, dtype=float)

    # Add epsilon to historically zero values to avoid strict division by zero crashes
    epsilon = 1e-10
    # Apply epsilon to P array
    p = p + epsilon
    # Apply epsilon to Q array
    q = q + epsilon

    # Normalize array P so everything sums precisely to 1.0
    p = p / np.sum(p)
    # Normalize array Q so everything sums precisely to 1.0
    q = q / np.sum(q)

    # Calculate actual cross-entropy distance
    return entropy(p, q)

# Define a function to calculate 1D Wasserstein distance
def calculate_wasserstein_distance(dist_a, dist_b):
    """
    Calculates the 1D Wasserstein distance (Earth Mover's Distance).
    Useful for comparing feature distributions (e.g. measuring data drift severity).
    If distributions are multi-dimensional, we calculate the average distance across dimensions.
    """
    # Convert distribution A to a numerical array
    dist_a = np.asarray(dist_a)
    # Convert distribution B to a numerical array
    dist_b = np.asarray(dist_b)

    # If the provided arrays are plain 1-dimensional arrays
    if dist_a.ndim == 1 and dist_b.ndim == 1:
        # Pass directly into scipy's algorithm
        return wasserstein_distance(dist_a, dist_b)
    
    # If the provided arrays are 2-dimensional (i.e. batches of features)
    if dist_a.ndim == 2 and dist_b.ndim == 2:
        # Prep an empty list to aggregate per-feature distance
        distances = []
        # Calculate per-feature distance and loop across columns (features)
        for i in range(dist_a.shape[1]):
            # Append the calculated 1D distance for this specific dimensional slice
            distances.append(wasserstein_distance(dist_a[:, i], dist_b[:, i]))
        # Take the aggregate mean of all distances to construct final single metric
        return np.mean(distances)

    # Hard error and crash out if 3D+ data format is unexpectedly passed
    raise ValueError("Input distributions must be 1D or 2D numpy arrays.")
