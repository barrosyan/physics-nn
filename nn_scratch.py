import numpy as np
import pickle

# Example 1: Neural Network for XOR Problem

# Activation function (ReLU)
def relu(x):
    return np.maximum(0, x)

# Derivative of the ReLU activation function
def relu_derivative(x):
    return (x > 0).astype(int)

# Activation function (Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the Sigmoid activation function
def sigmoid_derivative(x):
    return x * (1 - x)

# XOR Problem Input Data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Desired Output Labels for XOR Problem
y = np.array([[0], [1], [1], [0]])

# Neural Network Configuration
input_size = 2
hidden1_size = 4
hidden2_size = 4
output_size = 1
learning_rate = 0.1
epochs = 10000

# Weight and Bias Initialization
weights_input_hidden1 = np.random.uniform(size=(input_size, hidden1_size))
bias_hidden1 = np.zeros((1, hidden1_size))
weights_hidden1_hidden2 = np.random.uniform(size=(hidden1_size, hidden2_size))
bias_hidden2 = np.zeros((1, hidden2_size))
weights_hidden2_output = np.random.uniform(size=(hidden2_size, output_size))
bias_output = np.zeros((1, output_size))

# Training the Neural Network
for epoch in range(epochs):
    # Forward Propagation
    hidden1_input = np.dot(X, weights_input_hidden1) + bias_hidden1
    hidden1_output = relu(hidden1_input)
    hidden2_input = np.dot(hidden1_output, weights_hidden1_hidden2) + bias_hidden2
    hidden2_output = relu(hidden2_input)
    output_input = np.dot(hidden2_output, weights_hidden2_output) + bias_output
    output_output = sigmoid(output_input)
    
    # Error Calculation
    error = y - output_output
    
    # Gradient at the Output Layer
    delta_output = error * sigmoid_derivative(output_output)
    
    # Backpropagation at Hidden Layer 2
    error_hidden2 = delta_output.dot(weights_hidden2_output.T)
    delta_hidden2 = error_hidden2 * relu_derivative(hidden2_output)
    
    # Backpropagation at Hidden Layer 1
    error_hidden1 = delta_hidden2.dot(weights_hidden1_hidden2.T)
    delta_hidden1 = error_hidden1 * relu_derivative(hidden1_output)
    
    # Weight and Bias Updates
    weights_hidden2_output += hidden2_output.T.dot(delta_output) * learning_rate
    bias_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
    weights_hidden1_hidden2 += hidden1_output.T.dot(delta_hidden2) * learning_rate
    bias_hidden2 += np.sum(delta_hidden2, axis=0, keepdims=True) * learning_rate
    weights_input_hidden1 += X.T.dot(delta_hidden1) * learning_rate
    bias_hidden1 += np.sum(delta_hidden1, axis=0, keepdims=True) * learning_rate

# Testing the Neural Network on XOR Problem
test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
hidden1_input_test = np.dot(test_input, weights_input_hidden1) + bias_hidden1
hidden1_output_test = relu(hidden1_input_test)
hidden2_input_test = np.dot(hidden1_output_test, weights_hidden1_hidden2) + bias_hidden2
hidden2_output_test = relu(hidden2_input_test)
output_input_test = np.dot(hidden2_output_test, weights_hidden2_output) + bias_output
predicted_output_test = sigmoid(output_input_test)

print("Predicted Output:")
print(predicted_output_test)

# Example 2: Neural Network for CIFAR-10 Dataset

# CIFAR-10 Data Loading
def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def load_cifar10_data():
    # Loading training and test data
    train_data = []
    train_labels = []
    for i in range(1, 6):
        data_batch = unpickle(f'cifar-10-batches-py/data_batch_{i}')
        train_data.append(data_batch[b'data'])
        train_labels.extend(data_batch[b'labels'])
    train_data = np.vstack(train_data)
    test_data = unpickle('cifar-10-batches-py/test_batch')
    test_labels = test_data[b'labels']
    test_data = test_data[b'data']
    return train_data, np.array(train_labels), test_data, np.array(test_labels)

# Activation Function (ReLU)
def relu(x):
    return np.maximum(0, x)

# Derivative of the ReLU Activation Function
def relu_derivative(x):
    return (x > 0).astype(int)

# Activation Function (Softmax)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Data Loading and Preprocessing
X_train, y_train, X_test, y_test = load_cifar10_data()
X_train = X_train / 255.0
X_test = X_test / 255.0

# Neural Network Configuration
input_size = 3072  # 32x32x3
hidden1_size = 512
hidden2_size = 256
output_size = 10
learning_rate = 0.01
epochs = 50
batch_size = 128

# Weight and Bias Initialization
np.random.seed(0)
weights_input_hidden1 = np.random.randn(input_size, hidden1_size) / np.sqrt(input_size)
bias_hidden1 = np.zeros((1, hidden1_size))
weights_hidden1_hidden2 = np.random.randn(hidden1_size, hidden2_size) / np.sqrt(hidden1_size)
bias_hidden2 = np.zeros((1, hidden2_size))
weights_hidden2_output = np.random.randn(hidden2_size, output_size) / np.sqrt(hidden2_size)
bias_output = np.zeros((1, output_size))

# Training the Neural Network
for epoch in range(epochs):
    for i in range(0, X_train.shape[0], batch_size):
        X_mini = X_train[i:i + batch_size]
        y_mini = y_train[i:i + batch_size]
        
        # Forward Propagation
        hidden1_input = X_mini.dot(weights_input_hidden1) + bias_hidden1
        hidden1_output = relu(hidden1_input)
        hidden2_input = hidden1_output.dot(weights_hidden1_hidden2) + bias_hidden2
        hidden2_output = relu(hidden2_input)
        output_input = hidden2_output.dot(weights_hidden2_output) + bias_output
        output_output = softmax(output_input)
        
        # Error Calculation
        error = output_output
        error[range(len(y_mini)), y_mini] -= 1
        
        # Gradient at the Output Layer
        delta_output = error / batch_size
        
        # Backpropagation
        error_hidden2 = delta_output.dot(weights_hidden2_output.T)
        delta_hidden2 = error_hidden2 * relu_derivative(hidden2_input)
        error_hidden1 = delta_hidden2.dot(weights_hidden1_hidden2.T)
        delta_hidden1 = error_hidden1 * relu_derivative(hidden1_input)
        
        # Weight and Bias Updates
        weights_hidden2_output -= hidden2_output.T.dot(delta_output) * learning_rate
        bias_output -= np.sum(delta_output, axis=0, keepdims=True) * learning_rate
        weights_hidden1_hidden2 -= hidden1_output.T.dot(delta_hidden2) * learning_rate
        bias_hidden2 -= np.sum(delta_hidden2, axis=0, keepdims=True) * learning_rate
        weights_input_hidden1 -= X_mini.T.dot(delta_hidden1) * learning_rate
        bias_hidden1 -= np.sum(delta_hidden1, axis=0, keepdims=True) * learning_rate

    # Training Accuracy Calculation
    hidden1_input = X_train.dot(weights_input_hidden1) + bias_hidden1
    hidden1_output = relu(hidden1_input)
    hidden2_input = hidden1_output.dot(weights_hidden1_hidden2) + bias_hidden2
    hidden2_output = relu(hidden2_input)
    output_input = hidden2_output.dot(weights_hidden2_output) + bias_output
    output_output = softmax(output_input)
    predicted_labels = np.argmax(output_output, axis=1)
    accuracy = np.mean(predicted_labels == y_train)
    print(f"Epoch {epoch + 1}/{epochs}, Training Accuracy: {accuracy * 100:.2f}%")

# Evaluation of the Model on the Test Set
hidden1_input_test = X_test.dot(weights_input_hidden1) + bias_hidden1
hidden1_output_test = relu(hidden1_input_test)
hidden2_input_test = hidden1_output_test.dot(weights_hidden1_hidden2) + bias_hidden2
hidden2_output_test = relu(hidden2_input_test)
output_input_test = hidden2_output_test.dot(weights_hidden2_output) + bias_output
predicted_output_test = softmax(output_input_test)
predicted_labels_test = np.argmax(predicted_output_test, axis=1)
test_accuracy = np.mean(predicted_labels_test == y_test)
print(f"Test Set Accuracy: {test_accuracy * 100:.2f}%")

# Forward Propagation for Predictions
hidden1_input_test = X_test.dot(weights_input_hidden1) + bias_hidden1
hidden1_output_test = relu(hidden1_input_test)
hidden2_input_test = hidden1_output_test.dot(weights_hidden1_hidden2) + bias_hidden2
hidden2_output_test = relu(hidden2_input_test)
output_input_test = hidden2_output_test.dot(weights_hidden2_output) + bias_output
predicted_output_test = softmax(output_input_test)

# Displaying the First 10 Predictions and Real Labels
for i in range(10):
    print(f"Prediction: {predicted_labels_test[i]}, Real Label: {y_test[i]}")
