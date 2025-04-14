import numpy as np

# Step 1: Create network architecture
L = 1
n = [4, 1]  # Only input layer and output layer

print("layer 0 / input layer size", n[0])
print("layer 1 / output layer size", n[1])

# Step 2: Initialize weights and bias
W = np.random.randn(n[1], n[0])  # shape: (1, 4)
b = np.random.randn(n[1], 1)     # shape: (1, 1)

print("Weights shape:", W.shape)
print("Bias shape:", b.shape)

# Step 3: Create training data and labels
def prepare_data():
    X = np.array([
        [100, 200, 1, 60],   # OK
        [300, 500, 4, 90],   # At risk
        [120, 180, 0, 50],   # OK
        [400, 600, 7, 95],   # At risk
        [150, 250, 2, 65],   # OK
        [250, 450, 5, 85],   # At risk
        [110, 190, 0, 55],   # OK
        [350, 550, 6, 92],   # At risk
        [130, 220, 1, 62],   # OK
        [280, 480, 4, 88]    # At risk
    ])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    m = X.shape[0]
    A0 = X.T  # shape: (4, 10)
    Y = y.reshape(n[L], m)  # shape: (1, 10)
    return A0, Y

# Step 4: Activation function (step)
def step(arr):
    return np.where(arr > 0, 1, 0)

# Step 5: Feed forward process
def feed_forward(A0):
    Z = W @ A0 + b
    A = step(Z)
    return A

# Step 6: Training loop using Rosenblatt’s rule
def train(A0, Y, epochs=10, learning_rate=0.01):
    global W, b  # Allow updates to the global weights and bias
    m = A0.shape[1]  # Number of training examples

    for epoch in range(epochs):
        for i in range(m):
            # Extract the i-th training example and label
            x_i = A0[:, i].reshape(n[0], 1)   # Shape: (4, 1) — column vector for single input
            y_i = Y[0, i]                     # Shape: scalar — expected output (0 or 1)

            # Forward pass for a single example
            z_i = W @ x_i + b                 # Weighted sum (dot product + bias), shape: (1, 1)
            a_i = step(z_i)                   # Apply step activation → prediction (0 or 1), shape: (1, 1)

            # Calculate error between predicted and actual value
            error = y_i - a_i                 # If prediction is wrong, error will be ±1

            # Update weights and bias according to the perceptron learning rule:
            # w = w + learning_rate * error * input
            # b = b + learning_rate * error
            W += learning_rate * error * x_i.T  # x_i.T shape: (1, 4), so W shape stays (1, 4)
            b += learning_rate * error          # Adjust bias (broadcast to shape (1, 1))

        print(f"Epoch {epoch+1} complete")  # Notify when each epoch is done

# Call functions
A0, Y = prepare_data()
train(A0, Y, epochs=20, learning_rate=0.01)
y_hat = feed_forward(A0)

print("Predictions (y_hat):")
print(y_hat)
print("Actual (Y):")
print(Y)