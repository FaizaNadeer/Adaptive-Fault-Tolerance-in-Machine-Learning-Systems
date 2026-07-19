import numpy as np
import joblib
from sklearn.datasets import make_classification

# ============================================================
# PLUGIN IMPORTS
# ============================================================
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from self_healing_plugin.core.context import HealingContext
from self_healing_plugin.core.plugin import SelfHealingPlugin

from self_healing_plugin.detectors.simple_drift import SimpleMeanShiftDetector
from self_healing_plugin.detectors.overconfidence import OverconfidenceDetector
from self_healing_plugin.detectors.label_noise import LabelNoiseDetector

from self_healing_plugin.errors.error_classifier import RuleBasedErrorClassifier
from self_healing_plugin.policies.healing_policy import DefaultHealingPolicy
from self_healing_plugin.safety.data_guard import DataValidityGuard
from self_healing_plugin.adapters.sklearn_adapter import SklearnAdapter

# ============================================================
#DEFINE FUNCTIONS
# ============================================================

def clean_signals(signals):
    clean = {}
    for k, v in signals.items():
        if hasattr(v, "item"):      # numpy scalar
            clean[k] = v.item()
        else:
            clean[k] = v
    return clean

def log_event(ctx):
    print(
        f"[PLUGIN] error={ctx.error_type}, "
        f"result={ctx.result}, "
        f"signals={ctx.signals}",
        flush=True
    )
    
# ============================================================
# LOAD PRE-TRAINED MODEL
# ============================================================
model_holder = {
    "model": joblib.load("model.pkl")
}


# ============================================================
# REFERENCE TRAINING DATA (CLEAN)
# ============================================================
X_train, y_train = make_classification(
    n_samples=500,
    n_features=5,
    n_informative=3,
    random_state=42
)


# ============================================================
# STREAM DATA WITH DATA DRIFT (ERROR INJECTION)
# ============================================================
X_stream, y_stream = make_classification(
    n_samples=200,
    n_features=5,
    n_informative=3,
    random_state=1
)

# 🔴 Inject DATA DRIFT (mean shift)
X_stream = X_stream + 5.0


# ============================================================
# SET UP SELF-HEALING PLUGIN
# ============================================================
adapter = SklearnAdapter(model_holder)

plugin = SelfHealingPlugin(
    detectors=[
        SimpleMeanShiftDetector(threshold=0.1),          # DATA_DRIFT
        OverconfidenceDetector(entropy_threshold=0.3),   # OVERCONFIDENCE
        LabelNoiseDetector(disagreement_threshold=0.3),  # LABEL_NOISE
    ],
    error_classifier=RuleBasedErrorClassifier(),
    policy_engine=DefaultHealingPolicy(adapter),
    safety_guards=[DataValidityGuard()],
    logger=log_event
)


# ============================================================
# STEP 1 — NORMAL OPERATION (NO ERROR)
# ============================================================
print("\n--- STEP 1: Normal data (no error) ---", flush=True)

context_normal = HealingContext(
    model=model_holder["model"],
    X=X_train[:50],
    y=y_train[:50],
    metadata={
        "X_train": X_train,
        "y_train": y_train
    }
)

plugin.monitor(context_normal)


# ============================================================
# STEP 2 — DATA DRIFT (SELF-HEALING TRIGGERED)
# ============================================================
print("\n--- STEP 2: Data drift (self-healing) ---", flush=True)

context_drift = HealingContext(
    model=model_holder["model"],
    X=X_stream,
    y=y_stream,
    metadata={
        "X_train": X_train,
        "y_train": y_train
    }
)

plugin.monitor(context_drift)


# ============================================================
# STEP 3 — LABEL NOISE (BLOCKED FOR SAFETY)
# ============================================================
print("\n--- STEP 3: Label noise (blocked) ---", flush=True)

# 🔴 Inject LABEL NOISE (flip 40% of labels)
noisy_y = y_train.copy()
indices = np.random.choice(
    len(noisy_y),
    int(0.4 * len(noisy_y)),
    replace=False
)
noisy_y[indices] = 1 - noisy_y[indices]

context_label_noise = HealingContext(
    model=model_holder["model"],
    X=X_train[:200],
    y=noisy_y[:200],               # ← noisy labels passed here
    metadata={
        "X_train": X_train,
        "y_train": y_train
    }
)

plugin.monitor(context_label_noise)


# ============================================================
# FINAL STATE
# ============================================================
print("\n✅ Model after self-healing:", model_holder["model"])
