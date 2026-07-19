# Import numpy for numerical and statistical operations
import numpy as np
# Import GaussianProcessRegressor from sklearn for predictive variance modeling
from sklearn.gaussian_process import GaussianProcessRegressor
# Import appropriate kernels for modeling underlying distribution noise
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
# Import the abstract base class for safety guards
from self_healing_plugin.safety.base import SafetyGuard

# Define a guard that checks data uncertainty before allowing a retrain
class GPRUncertaintyGuard(SafetyGuard):
    """
    Evaluates the uncertainty of new incoming data before allowing retraining.
    If the data is too noisy (high predictive variance), it blocks the action.
    """
    # Initialize with a variance threshold limit
    def __init__(self, variance_threshold=0.8):
        # Store the defined limit
        self.variance_threshold = variance_threshold
        
        # A simple GPR with RBF and White kernel to explicitly capture data noise
        self.kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
        # Initialize the GPR model with the specified kernels
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, random_state=42)

    # Core logic determining if action is permitted to proceed
    def allow(self, context):
        """
        Calculates the predictive variance using Gaussian Process Regression.
        Returns False if the average variance exceeds the threshold, blocking the action.
        """
        try:
            # Import RetrainAction payload specifically to check the intended action
            from self_healing_plugin.actions.retrain import RetrainAction
            # If the action is anything other than retraining (like blocking or fallback), it's safe!
            if not isinstance(getattr(context, "action", None), RetrainAction):
                # Allow it to process without variance checks
                return True
                
            # We need the historical baseline data to fit the GPR to assess what 'normal' is
            X_train = context.metadata.get("X_train")
            # Fetch corresponding labels
            y_train = context.metadata.get("y_train")
            
            # If historical context isn't provided
            if X_train is None or y_train is None:
                # If we don't have baseline data, we can't estimate uncertainty. Allow safely.
                return True
                
            # For performance in streaming scenarios, we might only fit on a subset
            # Here we fit on the baseline data to train the GPR
            self.gpr.fit(X_train, y_train)
            
            # Predict the variance (uncertainty std_dev) on the newly acquired (drifted) data batch
            _, std_dev = self.gpr.predict(context.X, return_std=True)
            
            # Extract the actual mathematical average variance across the entire batch
            avg_variance = np.mean(std_dev ** 2)
            
            # Add this exact calculated scalar to signals so it gets logged by the EventRegistry
            context.signals["gpr_variance"] = float(avg_variance)
            # Log the boolean of whether it explicitly breached the limit
            context.signals["gpr_blocked"] = bool(avg_variance > self.variance_threshold)
            
            # Check if the calculated metric breaches the allowed limit
            if avg_variance > self.variance_threshold:
                # Print explicit warning to standard out for terminal monitoring
                print(f"[GPR_GUARD] Action BLOCKED. Data variance ({avg_variance:.2f}) exceeds threshold ({self.variance_threshold}).")
                # Return False to immediately halt the execution chain
                return False
                
            # If everything is within normal limits, allow the execution to continue
            return True
            
        except Exception as e:
            # Print a warning explaining the algorithmic failure
            print(f"[GPR_GUARD] Error during uncertainty calculation: {str(e)}. Defaulting to safe BLOCK.")
            # Default to blocking to prevent undefined or dangerous behaviors
            return False
