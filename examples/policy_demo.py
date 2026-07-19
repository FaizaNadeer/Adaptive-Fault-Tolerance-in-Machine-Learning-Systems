# self_healing_plugin/examples/policy_demo.py

# Import sys for system directory logic
import sys
# Import os for operating system pathing logic
import os

# --- Environment Setup ---
# Add the parent directory of self_healing_plugin to the Python path
# so we can import the package modules even if not installed via pip.
# This assumes the script is located in self_healing_plugin/examples/.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import SelfHealingPlugin engine core
from self_healing_plugin.core.plugin import SelfHealingPlugin
# Import basic Context data wrapper
from self_healing_plugin.core.context import HealingContext
# Import string converter for error enumeration states
from self_healing_plugin.errors.error_classifier import RuleBasedErrorClassifier
# Import custom configuration-driven strategy behavior generator
from self_healing_plugin.policies.configurable_policy import ConfigurableHealingPolicy

# --- Mock Components for Demonstration ---

# Create MockAdapter
class MockAdapter:
    """
    Simulates the orchestrator adapter.
    Instead of actually retraining a large model, it prints a success message.
    """
    # Overload update model parameters 
    def update_model(self, model):
        # Explicit successful terminal string payload 
        print("MockAdapter: Model updated!")

# Create MockDetector generating fixed fake payloads 
class MockDetector:
    """
    Simulates a drift detector.
    Always returns validation signals indicating drift with high confidence (low entropy).
    """
    # Expose basic detect implementation requirement
    def detect(self, context):
        # Simulate drift detection:
        # drift=True: A problem was found.
        # entropy=0.1: Very low uncertainty (high confidence) in this finding.
        return {"drift": True, "entropy": 0.1} 

# Create MockGuard implementing 100% false-pass checks
class MockGuard:
    """
    Simulates a safety guard (e.g., DataValidityGuard).
    Checks if the action is allowed. For this demo, it always approves.
    """
    # Expose fundamental 'allow' interface return rule
    def allow(self, context):
        # Dump testing alert line
        print("MockGuard: Checking safety...")
        # Blindly return successful parameter checking boolean
        return True

# Initialize generic custom logger
def logger(context):
    """
    Simple logger to print the final decision of the self-healing cycle.
    """
    # Print the error alongside mapping resulting logic path execution sequence enum
    print(f"Logger: {context.error_type} -> {context.result}")

# --- Main Execution Flow ---

# Standard execution loop routine block def
def main():
    # Header log output
    print("--- Policy Demo ---")
    
    # 1. Initialize Components
    # Instantiate standalone dummy action adapter payload operator
    adapter = MockAdapter()
    
    # Locate the policy configuration file.
    # It is expected to be in the parent directory of this script.
    config_path = os.path.join(os.path.dirname(__file__), "../policy_config.json")
    # System path validation print diagnostic
    print(f"Loading config from: {os.path.abspath(config_path)}")
    
    # Initialize the Configurable Policy Engine with the config file.
    policy_engine = ConfigurableHealingPolicy(adapter, config_path)
    
    # Initialize the Self-Healing Plugin with all its dependencies.
    plugin = SelfHealingPlugin(
        # Inject custom-built test mock component array list definition
        detectors=[MockDetector()],             # Use our mock detector
        # Load literal canonical String translator object constructor block
        error_classifier=RuleBasedErrorClassifier(), # Use standard rule-based classifier
        # Bind the loaded custom configurator dynamically mounted from file IO mapping 
        policy_engine=policy_engine,            # Use our loaded policy engine
        # Attach singleton array instance containing blindly approving mock guards test
        safety_guards=[MockGuard()],            # Use our mock guard
        # Direct event dispatch function stream listener pointer registration mechanism
        logger=logger                           # Use our print logger
    )
    
    # 2. Test Case 1: High Confidence Drift
    # Scenario: Detector finds drift with Entropy 0.1 (High Confidence).
    # Expected Policy: DATA_DRIFT -> RETRAIN (because confidence > 0.6).
    
    # Log the test instantiation block and expected parameters
    print("\nTest 1: High Confidence Drift (Entropy 0.1)")
    
    # Create the healing context.
    # We provide dummy training data (X_train, y_train) with 2 classes [0, 1]
    # to satisfy sklearn's LogisticRegression requirements in RetrainAction.
    context = HealingContext(
        # Not required for simple decision mapping block simulation evaluation parameter testing
        model=None, 
        # Feature placeholder parameters
        X=[[2]], 
        # Labels placeholder parameters
        y=[1], 
        # Signals initialized empty array instance mapping definition mapping
        signals={}, 
        # Minimal training meta required to bypass fundamental sklearn adapter requirements
        metadata={"X_train": [[0], [1]], "y_train": [0, 1]}
    )
    
    # Run the monitoring cycle.
    plugin.monitor(context)
    
    # 3. Test Case 2: Low Confidence Drift
    # Scenario: Detector finds drift with Entropy 0.8 (Low Confidence).
    # Expected Policy: DATA_DRIFT -> BLOCK (because confidence < 0.6).
    
    # Emit second testing evaluation print
    print("\nTest 2: Low Confidence Drift (Entropy 0.8)")
    
    # Define a bespoke detector for this specific test case
    class LowConfDetector:
        # Define core detector overriding loop mechanism return object
        def detect(self, context):
            # Return drift=True but high entropy (0.8).
            # Implied Confidence = 1.0 - 0.8 = 0.2
            # This is below the threshold of 0.6 in policy_config.json.
            return {"drift": True, "entropy": 0.8} 
            
    # Swap the internal array list of detectors live mapped into the overarching structure instance
    plugin.detectors = [LowConfDetector()]
    
    # Create a fresh context for the second run
    context2 = HealingContext(
        # Nil model
        model=None, 
        # Blank Data
        X=[[2]], 
        # Blank Response Vector
        y=[1], 
        # Fresh parameters
        signals={}, 
        # Retain bypass metadata instance parameter loading checks
        metadata={"X_train": [[0], [1]], "y_train": [0, 1]}
    )
    
    # Run the monitoring cycle.
    plugin.monitor(context2)

# Ensure scoping protection block against remote import collisions and double executions
if __name__ == "__main__":
    # Fire primary loop
    main()
