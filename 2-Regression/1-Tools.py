import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model, model_selection
from sklearn.metrics import mean_absolute_error, mean_squared_error

X, y = datasets.load_diabetes(return_X_y=True)

print(f"the shape of X: {X.shape}")

print("the first sample of X", X[0])


# only get the third column
X_col3 = X[:, 2]
print(f"shape of x_col3: {X_col3.shape}")
X_col3 = X_col3.reshape((-1, 1))
print(f"shape of x_col3: {X_col3.shape}")

# just use the third column feature to train
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X_col3, y, test_size=0.33
)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)


print(f"shape of y_test: {y_test.shape}, shape of y_pred: {y_pred.shape}")
print(f"mae: {mean_absolute_error(y_test, y_pred)}")
print(f"mse: {mean_squared_error(y_test, y_pred)}")

plt.scatter(X_test, y_test, c="blue")

# As the model is a Linear Model
# so the y_pred[i] = X_test[i] * weight + b
# Use them to draw a line plot directly
plt.plot(X_test, y_pred, c="red", linewidth=3)
plt.xlabel("Scaled BMIs")
plt.ylabel("Disease Progression")
plt.title("Graph Plot Showing Diabetes Progression Against BMI")
plt.savefig("./output/1.png")
