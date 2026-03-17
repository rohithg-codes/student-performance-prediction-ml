import numpy as np

# MODEL FUNCTION

def predict(X, weights, bias):
    return np.dot(X, weights) + bias


# TRAINING FUNCTION

def train_model(X, Y, lr=0.001, epochs=1000):
    n = len(X)
    weights = np.zeros(X.shape[1])
    bias = 0

    for i in range(epochs):
        y_pred = predict(X, weights, bias)

        # Gradients
        dw = (-2/n) * np.dot(X.T, (Y - y_pred))
        db = (-2/n) * np.sum(Y - y_pred)

        # Update parameters
        weights -= lr * dw
        bias -= lr * db

    return weights, bias

# DATA (REALISTIC)

X = np.array([
    [2, 6, 5],
    [3, 7, 4],
    [4, 6, 6],
    [5, 8, 3],
    [6, 7, 2],
    [7, 6, 2],
    [8, 8, 1],
    [9, 7, 1]
])

Y = np.array([40, 50, 55, 65, 70, 75, 85, 90])



# TRAIN MODEL

weights, bias = train_model(X, Y)

print("Model trained successfully!")
print("Weights:", weights)
print("Bias:", bias)


# USER INPUT

study = float(input("Enter study hours: "))
sleep = float(input("Enter sleep hours: "))
phone = float(input("Enter phone usage hours: "))

user_data = np.array([study, sleep, phone])

# Prediction
prediction = predict(user_data, weights, bias)

print("Predicted marks:", round(prediction, 2))