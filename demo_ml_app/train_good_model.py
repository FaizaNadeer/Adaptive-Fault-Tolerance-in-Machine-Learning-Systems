# Import joblib for model serialization saving
import joblib
# Import utility to artificially generate classification data
from sklearn.datasets import make_classification
# Import the baseline model algorithm
from sklearn.linear_model import LogisticRegression

# Train a GOOD model using synthetic datasets
X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    random_state=42
)

# Instantiate the basic Logistic Regression classifier
model = LogisticRegression()
# Train it on the pristine generated data
model.fit(X, y)

# Dump the finalized trained weights to disk for later use
joblib.dump(model, "model.pkl")
# Print user friendly confirmation statement
print("✅ Good model trained and saved")
