# Import numpy to handle standard matrix and data array capabilities
import numpy as np
# Import fake classification generator to mock ML flow
from sklearn.datasets import make_classification
# Import standard log-reg model for simple prediction proofs
from sklearn.linear_model import LogisticRegression

# Import central plugin context wrapper
from self_healing_plugin.core.context import HealingContext
# Import actual plugin overseer framework
from self_healing_plugin.core.plugin import SelfHealingPlugin

# Import standard mean difference calculator for drift determination
from self_healing_plugin.detectors.simple_drift import SimpleMeanShiftDetector
# Import rule enforcer that decides the classification of any drift detected
from self_healing_plugin.errors.error_classifier import RuleBasedErrorClassifier
# Import baseline static logic tree to connect anomaly states to actions
from self_healing_plugin.policies.healing_policy import DefaultHealingPolicy
# Import initial data validation safeguard
from self_healing_plugin.safety.data_guard import DataValidityGuard
# Import concrete sklearn wrapper class
from self_healing_plugin.adapters.sklearn_adapter import SklearnAdapter

# Data Generation Region
# Create 500 records of standard dataset logic
X, y = make_classification(n_samples=500, random_state=42)
# Treat the foremost 300 instances as the offline/historical starting zone
X_train, y_train = X[:300], y[:300]
# Treat the trailing 200 instances as an incoming "live" online transmission stream
X_stream = X[300:]
y_stream = y[300:]

# Initial model wrapping logic
# Create generic python dictionary proxy
model_holder = {}
# Execute standard gradient descent fit against the original clean 300 baseline
model = LogisticRegression().fit(X_train, y_train)
# Mount the compiled model inside the holder
model_holder["model"] = model

# Initiate the Sklearn adapter pointing at the encapsulated dictionary pointer
adapter = SklearnAdapter(model_holder)

# Plugin Construction Logic
plugin = SelfHealingPlugin(
    # Implement very rigid simple 0.1 threshold drift listener
    detectors=[SimpleMeanShiftDetector(threshold=0.1)],
    # Hardwire the standard Error Type mappings logic
    error_classifier=RuleBasedErrorClassifier(),
    # Feed default retrain/block logic without reading a JSON config
    policy_engine=DefaultHealingPolicy(adapter),
    # Preempt any dangerous operations using simple parameter rules 
    safety_guards=[DataValidityGuard()],
    # Provide one-line anonymous print function summarizing the event cycle
    logger=lambda ctx: print(
        f"[PLUGIN] error={ctx.error_type}, result={ctx.result}, signals={ctx.signals}"
    )
)

# Simulate streaming batches iterating exactly 5 times over chunks
for i in range(5):
    # Rip off segments dynamically sized at 20 rows, and artificially increment values to simulate drifting offset
    batch = X_stream[i*20:(i+1)*20] + i  # inject drift
    y_batch = y_stream[i*20:(i+1)*20]
    # Box the newly pulled stream content into context object format
    context = HealingContext(
        # Link current live production model wrapper instance pointer
        model=model_holder["model"],
        # Input newly aggregated feature array inputs
        X=batch,
        # Ground truth labels for the stream
        y=y_batch,
        # Enclose historic pristine training distributions into generic metadata vault for guards to reference
        metadata={
            "X_train": X_train,
            "y_train": y_train
        }
    )
    # Ping the master plugin to monitor, deduce, optionally heal, and dispatch signals based on findings
    plugin.monitor(context)
