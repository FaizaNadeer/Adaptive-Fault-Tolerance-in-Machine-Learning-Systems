# Import os for operating system path manipulation
import os
# Import sys to modify system paths for module discovery
import sys
# Import numpy for fast numerical computations
import numpy as np
# Import make_classification to generate artificial datasets
from sklearn.datasets import make_classification
# Import LogisticRegression as the base machine learning model
from sklearn.linear_model import LogisticRegression
# Import accuracy_score for evaluating model performance
from sklearn.metrics import accuracy_score
# Import warnings to suppress messy output logs
import warnings

# Suppress warnings to keep console output clean for the demo
warnings.filterwarnings('ignore')
# Append the parent directory to sys.path so the plugin module can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the core context object to pass data into the plugin
from self_healing_plugin.core.context import HealingContext
# Import the main Plugin orchestrator class
from self_healing_plugin.core.plugin import SelfHealingPlugin
# Import the localized event registry for viewing logs
from self_healing_plugin.core.registry import registry
# Import SimpleMeanShiftDetector to catch data translation offset issues
from self_healing_plugin.detectors.simple_drift import SimpleMeanShiftDetector
# Import RuleBasedErrorClassifier to convert drift signals into canonical string constants
from self_healing_plugin.errors.error_classifier import RuleBasedErrorClassifier
# Import ConfigDrivenHealingPolicy to act based on JSON configurations
from self_healing_plugin.policies.config_policy import ConfigDrivenHealingPolicy
# Import GPRUncertaintyGuard to block retraining on excessively noisy chaotic data
from self_healing_plugin.safety.uncertainty_guard import GPRUncertaintyGuard
# Import SklearnAdapter to allow the policy to edit the underlying model
from self_healing_plugin.adapters.sklearn_adapter import SklearnAdapter

# Define a custom printing function for logging events gracefully
def log_event(ctx):
    # Print the Final Decision made by the plugin including signals
    print(f"[PLUGIN FINAL DECISION] Error: {ctx.error_type} | Action: {ctx.result} | Signals: {ctx.signals}")

# Define the primary execution loop of this particular demo application
def main():
    # Print the decorative header
    print("=====================================================")
    # Print the demo title
    print("🛡️ Self-Healing Plugin - Advanced GPR Guard Demo")
    # Print closing decoration
    print("=====================================================\n")

    # 1. Generate synthetic dataset for baseline evaluation
    X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, random_state=42)
    # Split the dataset into 80% Training and 20% Validation feature blocks
    X_train, X_val = X[:800], X[800:]
    # Split the dataset into corresponding labels
    y_train, y_val = y[:800], y[800:]

    # 2. Train baseline model initialization message
    print("🏋️ Training initial baseline model...")
    # Initialize a pristine Logistic Regression model 
    model = LogisticRegression()
    # Fit the initial model to the clean training data
    model.fit(X_train, y_train)
    # Compute accuracy manually on the validation holdout split
    acc = accuracy_score(y_val, model.predict(X_val))
    # Print the resulting accuracy output
    print(f"✅ Baseline Accuracy: {acc:.2%}\n")

    # Wrap the model in a generic standard python dictionary to pass around easily
    model_holder = {"model": model}
    # Pass the dictionary wrapper into the specific adapter designed to handle sklearn instances
    adapter = SklearnAdapter(model_holder)

    # 3. Setup Plugin with the new GPR Guard printing message
    print("⚙️ Assembling Plugin with GPRUncertaintyGuard...")
    # Instantiate the grand central core of the healing machinery
    plugin = SelfHealingPlugin(
        # Register the array of mathematical anomaly detectors (just mean shift here)
        detectors=[SimpleMeanShiftDetector(threshold=0.5)], # Detects mean shifts
        # Set the classifier to classify those mathematical signals
        error_classifier=RuleBasedErrorClassifier(),
        # Point the decision making to the external 'policy_config.json' behavior tree
        policy_engine=ConfigDrivenHealingPolicy(adapter),
        # Enforce that safety guards must pass; if variance > 0.8, abort any retraining attempts
        safety_guards=[GPRUncertaintyGuard(variance_threshold=0.8)], # HIGH VARIANCE = BLOCK
        # Use our custom terminal printing logger above
        logger=log_event
    )

    # SCENARIO 1: Slight Drift (Safe to retrain)
    print("\n--- 🟢 SCENARIO 1: Slight Data Drift (Safe Learning) ---")
    # Shift the data slightly.
    X_slight_drift = X_val + 0.6 
    
    # Configure context passing incoming live slightly drifted features
    ctx1 = HealingContext(
        # Inject the active running model
        model=model_holder["model"],
        # Provide the new inputs
        X=X_slight_drift,
        # Provide ground truth answers for retroactive online learning validation
        y=y_val,
        # Supply historical baseline datasets to compare drift math against
        metadata={"X_train": X_train, "y_train": y_train}
    )
    
    # Expected: The drift detector catches a shift of 0.6.
    # The Error Classifier categorizes < 10.0 magnitude as SLIGHT_DRIFT.
    # The JSON config says SLIGHT_DRIFT = RETRAIN.
    # The GPR Guard checks the variance of the data. Since it's just a slight translation, variance is low.
    # Action ALLOWED -> RETRAIN runs -> SUCCESS.
    plugin.monitor(ctx1)


    # SCENARIO 2: Moderate Fault / Extreme Noise (FALLBACK Mechanism)
    print("\n--- 🔴 SCENARIO 2: Massive Sensor Shift (FALLBACK Retrieval) ---")
    # Add a massive directional shift to easily trigger MODERATE_FAULT (>10.0 magnitude)
    X_chaotic = X_val + 50.0 
    
    # Instantiate contexts wrapping the severely crippled incoming data
    ctx2 = HealingContext(
        # Provide active model
        model=model_holder["model"],
        # Input corrupted incoming data
        X=X_chaotic,
        # Pass labels
        y=y_val,
        # Pass memory/baseline
        metadata={"X_train": X_train, "y_train": y_train}
    )
    
    # Expected: The drift detector catches a MASSIVE shift.
    # The Error Classifier categorizes > 10.0 magnitude as MODERATE_FAULT.
    # The JSON config says MODERATE_FAULT = FALLBACK!
    # The FallbackAction will run NearestNeighbors on the historical safe baseline.
    plugin.monitor(ctx2)


    # 4. Check the Persistent Registry (Event Database)
    print("\n--- 💾 Checking Persistent Event Registry ---")
    # Query disk history via singleton registry, limited to the 2 runs just performed
    history = registry.get_history(limit=2)
    # Output number of recovered entries
    print(f"Found {len(history)} recent events logged to disk:")
    # Iterate across each fetched record in array list
    for i, event in enumerate(history):
        # Print the literal dumped string value of each occurrence sequentially
        print(f"  Event {i+1} [{event['timestamp']}]: Error={event['error_type']}, Action={event['action_result']}")

    # Print final completion decoration header
    print("\n=====================================================")
    # Print conclusion statement
    print("Demonstration Complete. 🎉")

# Standard boilerplate ensuring script execution logic fires only when called intrinsically
if __name__ == "__main__":
    # Ignite main runtime loop
    main()
