import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.DataFrame({
    "LOC": ["karen", "madaraka", "karen", "karen", "buruburu", "donholm", "langata", "langata", "donholm", "karen", "madaraka", "langata", "buruburu", "karen"],
    "FUR": ["yes", "yes", "no", "yes", "no", "no", "no", "yes", "yes", "yes", "yes", "no", "yes", "yes"],
    "AMB": ["serene", "semi_serene", "noisy", "semi_serene", "semi_serene", "serene", "very_noisy", "serene", "semi_serene", "serene", "noisy", "semi_serene", "semi_serene", "semi_serene"],
    "PROX_SCH": ["no", "yes", "no", "no", "no", "no", "yes", "no", "yes", "no", "yes", "yes", "no", "yes"],
    "PROX_ROAD": ["yes", "yes", "yes", "no", "yes", "no", "yes", "no", "no", "no", "yes", "no", "yes", "yes"],
    "PROX_MALL": ["yes", "yes", "yes", "no", "yes", "yes", "no", "yes", "no", "no", "no", "yes", "no", "yes"],
    "WATER": ["yes", "no", "yes", "yes", "yes", "no", "no", "yes", "yes", "no", "yes", "yes", "no", "yes"],
    "HK_SER": ["yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "no", "no", "yes", "yes", "no", "yes"],
    "SIZE": [32.50234527, 53.42680403, 61.53035803, 47.47563963, 59.81320787, 55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444, 45.41973014, 54.35163488, 44.1640495, 58.16847072],
    "PRICE": [31.70700585, 68.77759598, 62.5623823, 71.54663223, 87.23092513, 78.21151827, 79.64197305, 59.17148932, 75.3312423, 71.30087989, 55.16567715, 82.47884676, 62.00892325, 75.39287043]
})

# Extract SIZE as x and PRICE as y
x = data['SIZE'].values
y = data['PRICE'].values

# Define the Mean Squared Error function
def mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

# Define the Gradient Descent function
def gradient_descent(x, y, m, c, learning_rate):
    N = len(y)
    y_pred = m * x + c
    # Calculate gradients
    dm = (-2 / N) * np.sum(x * (y - y_pred))
    dc = (-2 / N) * np.sum(y - y_pred)
    # Update parameters
    m = m - learning_rate * dm
    c = c - learning_rate * dc
    return m, c

# Set initial values for m and c
m = np.random.rand()
c = np.random.rand()
learning_rate = 0.0001
epochs = 10

# Training the model
for epoch in range(epochs):
    y_pred = m * x + c
    mse = mean_squared_error(y, y_pred)
    print(f"Epoch {epoch+1}, Mean Squared Error: {mse:.4f}")
    m, c = gradient_descent(x, y, m, c, learning_rate)

# Plotting the line of best fit
plt.scatter(x, y, color="blue", label="Actual Data")
plt.plot(x, m * x + c, color="red", label="Line of Best Fit")
plt.xlabel("Office Size (sq. ft.)")
plt.ylabel("Office Price")
plt.title("Linear Regression - Line of Best Fit")
plt.legend()
plt.show()

# Predicting the office price for 100 sq. ft.
size_100 = 100
predicted_price = m * size_100 + c
print(f"The predicted office price for a size of 100 sq. ft. is: {predicted_price:.2f}")