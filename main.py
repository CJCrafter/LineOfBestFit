import numpy as np


def get_points_from_equation():
    def equation(x):
        return 3 * x + 4

    features = np.array([1, 2])
    return features, np.array([equation(x) for x in features])


# In machine learning "features" refer to the input variables
# and "target" refers to the "correct" or target answer.
features, target = get_points_from_equation()

# Usually we want numbers [-1, +1], but our dataset may contain numbers
# much larger than this. We use Z-Score Normalization to normalize our data
mean = np.mean(features)
std = np.std(features)
normalized_features = (features - mean) / std
features = normalized_features

# We use machine learning terms "weight" and "bias" instead of
# "m" (for slope) and "b" (for y-intercept). This is y=mx+b.
weight = 0
bias = 0


def predict(input):
    """Returns the prediction of our model"""
    return weight * input + bias


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
iterations = 500
learning_rate = 0.1
print(f"{'#':^5} | {'weight':^20} | {'bias':^20} | {'cost':^20}")
for i in range(iterations):

    # Taking the derivative of the cost function with respect to weight/bias
    size = len(features)
    weight_derivative = sum([(weight * features[i] + bias - target[i]) * features[i] for i in range(size)]) / size
    bias_derivative = sum([weight * features[i] + bias - target[i] for i in range(size)]) / size

    weight -= learning_rate * weight_derivative
    bias -= learning_rate * bias_derivative

    # This statement will print 10 updates throughout the loop
    if i % (iterations / 10) == 0:
        print(f"{i:<5} | {weight:<20} | {bias:<20} | {cost():<20}")

# Convert the normalized weight and bias back to their human-readable scale
weight_human_readable = weight / std
bias_human_readable = -weight * mean / std + bias

print(f"Found weight={weight_human_readable}, bias={bias_human_readable} by gradient descent")
