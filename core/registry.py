# Import json for serializing events to disk
import json
# Import os for path logic
import os
# Import time for timestamp generation
import time
# Import datetime for human readable iso formatting
from datetime import datetime

# Define EventRegistry class for persistent observability
class EventRegistry:
    """
    A lightweight, persistent logger for the Self-Healing Plugin.
    Records monitoring decisions, caught errors, signals, and actions taken
    into a local JSON-lines file.
    """
    # Initialize the registry, defaulting the log_dir
    def __init__(self, log_dir=None):
        # If no directory is provided
        if log_dir is None:
            # Set the default directory to the root 'logs' folder
            log_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "logs"
            )
        # Store the directory path
        self.log_dir = log_dir
        # Ensure the directory actually exists, creating it if not
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Define the absolute path to the json-lines database file
        self.log_file = os.path.join(self.log_dir, "healing_events.jsonl")

    # Define the core logging method
    def log_event(self, context):
        """
        Logs a single plugin monitor event to the persistent storage.
        """
        # Convert signals (which might contain numpy arrays/bools) to native Python types
        clean_signals = {}
        # Loop through key, value pairs in the context signals
        for k, v in context.signals.items():
            # Check if the value is a numpy datatype with .item() access
            if hasattr(v, "item"): 
                # Convert numpy specific types back to Python native
                clean_signals[k] = v.item()
            # Otherwise 
            else:
                # Store the value as is
                clean_signals[k] = v

        # Construct the event structure dictionary
        event_data = {
            # Attach a standardized ISO 8601 timestamp
            "timestamp": datetime.fromtimestamp(time.time()).isoformat(),
            # Convert error type to string cleanly
            "error_type": str(context.error_type),
            # Convert result output to string cleanly
            "action_result": str(context.result),
            # Attach the sanitized native python signals
            "signals": clean_signals
        }

        # Try to execute disk I/O safely
        try:
            # Open the file in Append mode
            with open(self.log_file, "a") as f:
                # Write the JSON blob and a newline
                f.write(json.dumps(event_data) + "\n")
        # Catch any file system permissions access errors
        except Exception as e:
            # Print a warning but do not crash the application
            print(f"Warning: Failed to write to EventRegistry: {e}")

    # Define a method to read historical data
    def get_history(self, limit=100):
        """
        Retrieves the most recent healing events.
        """
        # Check if the disk file actually exists
        if not os.path.exists(self.log_file):
            # Return an empty list if no logs exist
            return []
            
        # Initialize empty list to hold events
        events = []
        # Try to read the file safely
        try:
            # Open the file in Read mode
            with open(self.log_file, "r") as f:
                # Iterate line by line
                for line in f:
                    # Strip whitespace to check for empty lines
                    if line.strip():
                        # Unpack JSON and append to event list
                        events.append(json.loads(line))
        # Handle read errors
        except Exception as e:
            # Print a warning without crashing
            print(f"Error reading from registry: {e}")
            
        # Return only the most recent N elements defined by limit
        return events[-limit:]

# Singleton instance available for the whole plugin to import freely
registry = EventRegistry()
