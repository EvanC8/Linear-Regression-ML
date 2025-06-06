from model import LinearRegressionModel
from sklearn.model_selection import train_test_split
import numpy as np

"""MODEL FUNCTIONALITY DEMONSTRATION AND TESTS"""

# Setup test data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
true_weights = 2.5
bias = 7
noise = np.random.randn(100, 1)
y = true_weights * X + bias + noise

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = LinearRegressionModel(alpha=0.01, epochs=1000, track_metrics=True)

# Train model with metrics
model.fit(X_train, y_train, X_test, y_test)

# Get learned bias and weights
model_theta = model.get_params()["theta"]
print(f"Learned bias: {model_theta[0]}")
print(f"Learned weight: {model_theta[1]}")

# Get tracked train and test metrics
model_metrics = model.get_training_history()
print(f"Train and test metrics: {model_metrics}")

# Make predictions
model_predictions = model.predict(X_test)
print(f"Predictions: {model_predictions}")