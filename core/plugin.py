# self_healing_plugin/core/plugin.py

class SelfHealingPlugin:
    # The main class that orchestrates the self-healing process.
    # It coordinates detectors, classifiers, policies, and actions.
    
    def __init__(
        self,
        detectors,        # List of detector instances to monitor data signals
        error_classifier, # Component to classify detected signals into an ErrorType
        policy_engine,    # Component to decide the action based on ErrorType and signals
        safety_guards,    # List of guards to enforce safety constraints before execution
        logger            # Callable to log the context and results
    ):
        # Store the list of detectors (e.g., DriftDetector, OverconfidenceDetector)
        self.detectors = detectors
        
        # Store the error classifier strategy (e.g., RuleBasedErrorClassifier)
        self.error_classifier = error_classifier
        
        # Store the policy engine (e.g., ConfigurableHealingPolicy) which decides actions
        self.policy_engine = policy_engine
        
        # Store safety guards (e.g., DataValidityGuard) to prevent unsafe actions
        self.safety_guards = safety_guards
        
        # Store the logger for observability
        self.logger = logger

    def monitor(self, context):
        # Main entry point for monitoring a single inference or batch context.
        # context: HealingContext object containing model, data, and metadata.

        # 1. Detection Phase: Run all registered detectors
        for detector in self.detectors:
            # Detectors analyze the context and return a dictionary of signals (e.g., {"drift": True})
            # We update the context's signals with these findings.
            context.signals.update(detector.detect(context))

        # 2. Classification Phase: Determine the high-level error type
        # The classifier looks at the accumulated signals and decides the primary error (e.g., DATA_DRIFT).
        context.error_type = self.error_classifier.classify(context.signals)
        
        # 3. Decision Phase: Ask the policy engine what to do
        # The policy engine uses the error type and signals (for confidence checks) to return an Action.
        # Action is an instance of a class like RetrainAction or BlockAction.
        action = self.policy_engine.decide(context.error_type, context.signals)
        context.action = action

        # 4. Safety Phase: Verify the context with all safety guards
        for guard in self.safety_guards:
            # If any guard denies the action/context, we block immediately.
            if not guard.allow(context):
                # Mark result as BLOCKED
                context.result = "BLOCKED"
                # Log the blocked event
                self.logger(context)
                from self_healing_plugin.core.registry import registry
                registry.log_event(context)
                # Return context immediately, aborting the action
                return context

        # 5. Execution Phase: Execute the decided action
        # If we passed all guards, run the action (e.g., retrain the model).
        context.result = action.execute(context)
        
        # Log the final result of the cycle
        self.logger(context)
        from self_healing_plugin.core.registry import registry
        registry.log_event(context)
        
        # Return the updated context
        return context
