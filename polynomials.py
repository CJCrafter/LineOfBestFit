import numpy as np


def get_points_from_equation():
    def equation(x):
        return 10*x**2 + 2*x + 88

    features = np.array([0, 1, 2])
    return features, np.array([equation(x) for x in features])


# In machine learning "features" refer to the input variables
# and "target" refers to the "correct" or target answer.
features, target = get_points_from_equation()

# When degree=2, we will go up to x^2. Increasing the degree adds more
# features, which might make your model more accurate (but harder to train)
degree = 2
features = np.array([features**i for i in range(1, degree+1)]).T

# Usually we want numbers [-1, +1], but our dataset may contain numbers
# much larger than this. We use Z-Score Normalization to normalize our data
def normalize_features(features):
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    normalized_features = (features - means) / stds
    return normalized_features, means, stds


normalized_features, means, stds = normalize_features(features)
features = normalized_features

# We use machine learning terms "weight" and "bias" instead of
# "m" (for slope) and "b" (for y-intercept). This is y=mx+b.
weights = np.zeros(degree)
bias = 0


def predict(input):
    """Returns the prediction of our model"""
    return np.dot(weights, input) + bias


def cost():
    """Returns a grade for our model, bigger numbers means our model is doing poorly"""
    sum = 0
    size = len(features)

    # Accumulates the squared difference between the model's predictions
    # and what the target value is.
    for i in range(size):
        x = features[i]
        y_predicted = predict(x)
        y_actual = target[i]
        sum += (y_predicted - y_actual) ** 2

    sum /= size
    return sum


# Naturally, we want to reduce the "cost" of our model to the absolute minimum
# To minimize a function, we take the derivative of the cost function and use
# a gradient descent algorithm to "roll the ball down the hill" to the minimum
iterations = 1000
learning_rate = 0.1
header = [f"weight[{i}]" for i in range(degree)]
header = " | ".join([f'{string:<20}' for string in header])

print(f"{'#':^5} | {header} | {'bias':^20} | {'cost':^20}")
for i in range(iterations):

    # Taking the derivative of the cost function with respect to weight/bias
    size = len(features)
    weight_derivative = np.zeros(degree)
    for j in range(degree):
        weight_derivative[j] = sum([(predict(features[k]) - target[k]) * features[k][j] for k in range(size)]) / size
    bias_derivative = sum([predict(features[k]) - target[k] for k in range(size)]) / size

    # This is kindof like Euler's method... we are adding the "velocity" to the
    # "position" to calculate the "next position" (but replace motion terms with AI and cost function)
    weights -= learning_rate * weight_derivative
    bias -= learning_rate * bias_derivative

    # This statement will print 10 updates throughout the loop
    if i % (iterations / 10) == 0:
        weights_string = " | ".join([f"{weights[j]:<20}" for j in range(degree)])
        print(f"{i:<5} | {weights_string} | {bias:<20} | {cost():<20}")

# Convert the normalized weight and bias back to their human-readable scale
weight_human_readable = weights / stds
bias_human_readable = bias - np.sum((weights * means) / stds)

print(f"Found weight={weight_human_readable}, bias={bias_human_readable} by gradient descent")
