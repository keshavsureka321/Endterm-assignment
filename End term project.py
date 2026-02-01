import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# 1. Load the data
df = pd.read_csv("C:\\Users\\Keshav\\Downloads\\data.csv")
X = df['SquareFeet'].values
y = df['Price'].values

# 2. Feature Scaling (Standardization)
# Scaling is crucial for Gradient Descent to work effectively with different ranges
X_mean, X_std = np.mean(X), np.std(X)
y_mean, y_std = np.mean(y), np.std(y)

X_scaled = (X - X_mean) / X_std
y_scaled = (y - y_mean) / y_std

# Adding a column of ones for the intercept term (theta0)
# This allows us to use matrix multiplication: X_b . theta
X_b = np.c_[np.ones((len(X_scaled), 1)), X_scaled]

# 3. Linear Regression from Scratch (Gradient Descent)
learning_rate = 0.01
n_iterations = 1000
m = len(y_scaled)
theta = np.random.randn(2, 1)  # Initialize random weights
y_scaled = y_scaled.reshape(-1, 1)

cost_history = []

for iteration in range(n_iterations):
    # Calculate gradients
    # Gradient of MSE: (2/m) * X^T * (X*theta - y)
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y_scaled)

    # Update theta
    theta = theta - learning_rate * gradients

    # Record cost for visualization
    cost = np.mean((X_b.dot(theta) - y_scaled) ** 2)
    cost_history.append(cost)

# 4. Preparing results for visualization
# Create a range of X values for the prediction line
X_new = np.array([[X.min()], [X.max()]])
X_new_scaled = (X_new - X_mean) / X_std
X_new_b = np.c_[np.ones((2, 1)), X_new_scaled]

# Predict and scale back to original units
y_predict_scaled = X_new_b.dot(theta)
y_predict = y_predict_scaled * y_std + y_mean

# 5. Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X_new, y_predict, color='red', linewidth=3, label='Regression Line')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.title('Linear Regression: Square Feet vs Price')
plt.legend()
plt.savefig('regression_plot.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(n_iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Gradient Descent Convergence')
plt.savefig('cost_history.png')
plt.show()



