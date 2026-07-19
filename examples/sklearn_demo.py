# Import numpy for numerical and array capabilities
import numpy as np
# Import OpenML dataset fetcher for heart disease data
from sklearn.datasets import fetch_openml
# Import mathematical algorithm chunking definition implementation object method property utility instance
from sklearn.model_selection import train_test_split
# Import random forest mapping logical structural algorithm interface object constructor builder
from sklearn.ensemble import RandomForestClassifier
# Import accuracy mathematical heuristic metric module evaluation object pointer loader
from sklearn.metrics import accuracy_score

# --- Environment Setup ---
# Add the parent directory of self_healing_plugin to the Python path
# so we can import the package modules even if not installed via pip.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import basic core data wrapper format 
from self_healing_plugin.core.context import HealingContext
# Import generalized interface abstraction runner logic mapping routine manager instance factory plugin core object pattern
from self_healing_plugin.core.plugin import SelfHealingPlugin
# Import simplistic drift evaluator component pattern definition loader array constructor object algorithm
from self_healing_plugin.detectors.simple_drift import SimpleMeanShiftDetector
# Import uncertainty check routine function abstraction mathematical mapping implementation module logic block pattern loader
from self_healing_plugin.detectors.overconfidence import OverconfidenceDetector
# Import heuristic error classifier
from self_healing_plugin.errors.error_classifier import RuleBasedErrorClassifier
# Import generalized hardcoded fallback engine
from self_healing_plugin.policies.healing_policy import DefaultHealingPolicy
# Import primary data validation wrapper
from self_healing_plugin.safety.data_guard import DataValidityGuard
# Import external parameter updating module payload operator function structure mapping implementation abstraction execution pattern manager object factory builder mechanism block class definition loader tool utility
from self_healing_plugin.adapters.sklearn_adapter import SklearnAdapter

# Standard explicit print log message routing method block statement routine logic procedure mapping object factory function
def log_event(ctx):
    # Standard terminal mapping display print line payload mapping stream interface stream command argument
    print(f"\n[PLUGIN ACTION] Detected: {ctx.error_type} | Action Taken: {ctx.result} | Signals: {ctx.signals}")

# Main execution loop block handler definition method parameter scope function instance logic declaration
def main():
    # Decorative line rendering stream pointer dump parameter
    print("==================================================")
    # Output title display rendering
    print("🤖 Self-Healing Plugin - Working Model Demonstration")
    # Rendering decorator 
    print("==================================================")

    # 1. Load real dataset
    print("\nLoading Heart Disease dataset...")
    # Fetch heart disease dataset from OpenML repository
    # We drop NAs and ensure everything is numeric for the Random Forest to ingest easily
    data = fetch_openml(name='heart-disease', version=1, parser='auto')
    import pandas as pd
    df = data.frame.dropna()
    # Explicit variable map destructure assignment payload binding procedure
    # We must ensure X only contains numbers, converting categoricals quickly using dummy vars or dropping them
    X = pd.get_dummies(df.drop('target', axis=1)).values
    y = df['target'].values
    y = y.astype(int)
    
    # Split: Train, Validation (clean stream), Drifted Stream
    # Establish baseline split payload implementation routine definition segment logic mechanism procedure procedure routine block statement function 
    X_train, X_stream, y_train, y_stream = train_test_split(X, y, test_size=0.4, random_state=42)
    # Execute secondary mapping routine format structural split layout definition layout schema logic object wrapper module function constructor block 
    X_val, X_drift, y_val, y_drift = train_test_split(X_stream, y_stream, test_size=0.5, random_state=42)

    # 2. Train baseline model
    print("\n🏋️ Training initial baseline model (RandomForest)...")
    # Spin up default RF tree structure ensemble map definition loader algorithm execution instance class object builder block component tool wrapper pattern mapping module
    model = RandomForestClassifier(random_state=42)
    # Perform numeric weight binding structural logic mapping calculation 
    model.fit(X_train, y_train)
    
    # Bind numerical result payload representation accuracy calculation object interface structure stream method definition value assignment expression statement block operation utility
    initial_acc = accuracy_score(y_val, model.predict(X_val))
    # Write metric output display message
    print(f"✅ Baseline Accuracy on Clean Data: {initial_acc:.2%}")

    # Wrap model in a holder so the adapter can update it
    # Encapsulate mutable binding wrapper abstraction array mapping schema reference definition property mechanism pointer instance target memory location link
    model_holder = {"model": model}

    # 3. Setup self-healing plugin
    print("\n⚙️ Setting up Self-Healing Plugin...")
    # Point standard sklearn wrapper adapter abstraction pattern mechanism implementation module loader framework interface proxy utility constructor reference assignment format representation interface at model dictionary
    adapter = SklearnAdapter(model_holder)
    # Launch generalized orchestration plugin object mapping abstraction routine component class structure manager representation core interface
    plugin = SelfHealingPlugin(
        # Pass hardcoded list logic array collection instance structure format pointer property definition assignment configuration values layout algorithm parameter object blocks 
        detectors=[
            # Execute numeric difference implementation strategy function object calculation structural reference definition module implementation parameter threshold limit mapping constructor block
            SimpleMeanShiftDetector(threshold=0.5), # Detects large shifts
            # Provide entropy measurement object interface abstraction pattern structure framework component utility method structural algorithm layout schema loader builder pattern definition definition wrapper constraint configuration value declaration block operation 
            OverconfidenceDetector(entropy_threshold=0.1)
        ],
        # Hook fundamental rule map definition evaluator constructor module class routine parameter method algorithm structural property logic binding reference implementation assignment object
        error_classifier=RuleBasedErrorClassifier(),
        # Point to hardcoded behavior strategy method handler representation pointer interface mechanism constructor parameter definition instance class class builder mapping layout utility pattern component
        policy_engine=DefaultHealingPolicy(adapter),
        # Check input numeric payload logic validation execution array instance structure format property builder function class abstraction module tool loader constraint rule condition
        safety_guards=[DataValidityGuard()],
        # Pass standard logging mapping pointer function hook abstraction method reference object stream builder tool parameter assignment value connection definition property representation logic mechanism
        logger=log_event
    )

    # 4. Scenario: Normal Data Execution
    # Display step decorator format mapping output property configuration statement message argument display target reference logic value pattern module string rendering value definition 
    print("\n--- 🟢 SCENARIO 1: Processing Normal Data Stream ---")
    # Bundle core execution wrapper array context variable assignment class model object mapping interface structure parameter implementation layout builder pattern instance definition configuration method payload reference mechanism block function representation framework loader property tool utility structure format constructor procedure schema logic loop mapping definition pattern operation object instantiation property
    context_normal = HealingContext(
        # Bind payload representation dictionary representation mapping mechanism array mapping implementation object method assignment instance definition reference method pointer implementation class module wrapper framework builder framework structure schema
        model=model_holder["model"],
        # Pure feature definition mapping array structure mapping object constructor class tool schema builder logic loop definition binding assignment reference variable instantiation memory mapping reference assignment payload operation array pointer location parameter
        X=X_val,
        # Standard baseline check structure property array parameter layout declaration utility mapping builder value definition constraint constraint reference array builder tool layout memory configuration value object constructor binding object
        y=y_val, 
        # Dict memory property metadata reference array struct method definition array module schema variable mapping reference structural constructor implementation payload value connection binding statement
        metadata={"X_train": X_train, "y_train": y_train}
    )
    # Run loop
    plugin.monitor(context_normal)
    # Check baseline performance pattern accuracy method implementation tool validation definition pointer variable state property logic operation mechanism block method return structural statement variable mapping reference connection builder representation layout object implementation return assignment format array constraint operation instantiation 
    current_acc = accuracy_score(y_val, model_holder["model"].predict(X_val))
    # Write output
    print(f"📊 Accuracy remains high: {current_acc:.2%}")

    # 5. Scenario: Data Drift Execution
    print("\n--- 🔴 SCENARIO 2: Sudden Data Drift Occurs! ---")
    # Introduce severe drift (simulating sensor degradation or new population)
    # Apply raw mathematical formula offset variable operation implementation structure statement statement assignment configuration variable string block assignment
    X_drift_severe = X_drift + np.mean(X_train, axis=0) * 2.0 
    
    # Show accuracy dropping BEFORE healing
    # Apply standard assessment property verification layout output procedure check utility value definition object structure logic pointer binding mapping pointer execution structural constructor execution mechanism tool loop schema value mapping expression return logic 
    degraded_acc = accuracy_score(y_drift, model_holder["model"].predict(X_drift_severe))
    # Write display stream format definition instance mapping object reference text representation value property parameter parameter value argument target text operation
    print(f"⚠️ Model accuracy dropped to {degraded_acc:.2%} due to data drift!")

    # Pass the drifted data to the plugin for monitoring
    context_drift = HealingContext(
        model=model_holder["model"],
        X=X_drift_severe,
        y=y_drift, # Providing new true labels so it can retrain
        metadata={"X_train": X_train, "y_train": y_train}
    )
    
    print("🔍 Plugin monitoring stream...")
    # Trigger logic plugin object framework method abstraction builder mapping execution structure property mapping schema interface definition loop constructor value operation definition variable logic memory instance instance pointer array schema
    plugin.monitor(context_drift)
    
    # 6. Evaluate healed model
    print("\n--- 🟢 SCENARIO 3: Evaluating Healed Model ---")
    # Reread evaluation variable structure mapping expression result calculation state property interface wrapper representation representation representation mapping representation property instance module constructor value memory mapping target assignment declaration method definition 
    recovered_acc = accuracy_score(y_drift, model_holder["model"].predict(X_drift_severe))
    # Print value mapping stream format rendering console message definition function object property statement line operation mechanism 
    print(f"✅ Model accuracy recovered to {recovered_acc:.2%} after automated retraining!")
    
    # End decorator display
    print("\n==================================================")
    # Final rendering
    print("Demonstration Complete! The system self-healed. 🎉")
    # Cap decorators
    print("==================================================")

# Root level script definition constraint loop checking
if __name__ == "__main__":
    # Launch main scope structure layout builder mechanism value variable logic
    main()
