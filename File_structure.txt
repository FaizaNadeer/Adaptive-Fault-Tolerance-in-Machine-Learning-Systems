self_healing_plugin/
│
├── core/
│   ├── plugin.py              # Main plugin entry point
│   ├── context.py             # Shared runtime context
│   ├── registry.py            # Event & decision logging
│
├── detectors/
│   ├── base.py                # Detector interface
│   ├── gpr_drift.py           # Example detector
│
├── errors/
│   ├── error_types.py         # Error taxonomy
│   ├── error_classifier.py    # Error classification logic
│
├── policies/
│   ├── base.py                # Policy interface
│   ├── healing_policy.py      # Default policy engine
│
├── actions/
│   ├── base.py                # Healing action interface
│   ├── retrain.py             # Retrain action
│   ├── block.py               # Block / no-op action
│
├── safety/
│   ├── base.py                # Safety guard interface
│   ├── data_guard.py          # Data validity guard
│   ├── uncertainty_guard.py   # Uncertainty guard
│
├── adapters/
│   ├── orchestrator_adapter.py  # v7-style state machine wrapper
│
├── utils/
│   ├── metrics.py
│   ├── entropy.py
│
├── examples/
│   └── sklearn_demo.py
│
└── README.md
