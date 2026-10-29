import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

pumpkins = pd.read_csv("../data/US-pumpkins.csv")
# shape
# (1757, 26)

# just get all packages contains "bushel"
# (415, 26)
pumpkins = pumpkins[pumpkins["Package"].str.contains("bushel", case=True, regex=True)]

columns_to_select = [
    "Package",
    "Variety",
    "City Name",
    "Low Price",
    "High Price",
    "Date",
]

new_pumpkins = pumpkins.loc[:, columns_to_select]

price = (pumpkins["Low Price"] + pumpkins["High Price"]) / 2

month = pd.DatetimeIndex(pumpkins["Date"]).month
day_of_year = pd.to_datetime(pumpkins["Date"]).apply(
    lambda dt: (dt - datetime(dt.year, 1, 1)).days
)

new_pumpkins = pd.DataFrame(
    {
        "Month": month,
        "DayOfYear": day_of_year,
        "Variety": pumpkins["Variety"],
        "City": pumpkins["City Name"],
        "Package": pumpkins["Package"],
        "Low Price": pumpkins["Low Price"],
        "High Price": pumpkins["High Price"],
        "Price": price,
    }
)

# normalize the price
new_pumpkins.loc[new_pumpkins["Package"].str.contains("1 1/9"), "Price"] = price / (
    1 + 1 / 9
)
new_pumpkins.loc[new_pumpkins["Package"].str.contains("1/2"), "Price"] = price / (1 / 2)

# print basic information and correlations between datetime and price

print(new_pumpkins.head())
print(
    f"the correlation between month and price: {new_pumpkins['Month'].corr(new_pumpkins['Price']):.3f}"
)
print(
    f"the correlation between day of year and price: {new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']):.3f}"
)


# draw a scatter graph
# use different color for different Variety types

# base Axes
ax: plt.Axes = None
colors = ["red", "blue", "green", "yellow"]

for i, var in enumerate(new_pumpkins["Variety"].unique()):
    df = new_pumpkins[new_pumpkins["Variety"] == var]
    ax = df.plot.scatter("DayOfYear", "Price", ax=ax, c=colors[i], label=var)

fig = ax.get_figure()
fig.savefig("./output/7.png")

ax = new_pumpkins.groupby("Variety")["Price"].mean().plot(kind="bar")
fig = ax.get_figure()
fig.savefig("./output/8.png")

pie_pumpkins = new_pumpkins[new_pumpkins["Variety"] == "PIE TYPE"]
ax = pie_pumpkins.plot.scatter("DayOfYear", "Price")
fig = ax.get_figure()
fig.savefig("./output/9.png")

# calcualte the correlation between DayOfYear and the Price
print(
    f"the correlation between DayOfYear and Price on pie pumpkins: {pie_pumpkins['DayOfYear'].corr(pie_pumpkins['Price']):.3f}"
)
# about -0.26
# it make sense to train a model based on the DayOfYear feature
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Import scikit-learn models
# use simple feature as input
# so we should use reshape(-1, 1) to convert this one dimension input into two dimension
X = pie_pumpkins["DayOfYear"].to_numpy().reshape(-1, 1)
y = pie_pumpkins["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

# train linear regression model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# rmse is the root mean squart error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:3.3} ({rmse / np.mean(y_pred) * 100:3.3}%)")

score = model.score(X_train, y_train)
print("Model determination: ", score)

ax = plt.axes()
ax.scatter(X_test, y_test, c="blue")
ax.plot(X_test, y_pred, c="red")
fig: plt.Figure = ax.get_figure()
fig.savefig("./output/10.png")


# polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# PolynomialFeatures(2) means the max power of feature combination will be 2
# for two features it will be x^2, y^2, xy, x, y
#
# for single feature, it will be x^2, x
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:3.3} ({rmse / np.mean(y_pred) * 100:3.3}%)")

score = pipeline.score(X_train, y_train)
print("Model determination: ", score)

# WARN:
# You should not draw this figure directly
# as the curve is not a linear plot
# create more points on curve instead
#
# ax = plt.axes()
# ax.scatter(X_test, y_test, c="blue")
# ax.plot(X_test, y_pred, c="red")

# must reshape this because input for linear regression should be 2-D
# pick 100 points, the curve will be smooth
X_range = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
y_range = pipeline.predict(X_range)


ax = plt.axes()
ax.scatter(X_test, y_test, c="blue")
ax.plot(X_range, y_range, c="red")
fig: plt.Figure = ax.get_figure()
fig.savefig("./output/11.png")


# NOTE: Convert category into number so that it can be processed by linear models

# One-shot encoding create a new table with category columns
# if this item belongs to this category, the value will be True else False

print(pd.get_dummies(new_pumpkins["Variety"]))

# to concate different columns with each other, we can use the join function

X = (
    pd.get_dummies(new_pumpkins["Variety"])
    .join(new_pumpkins["Month"])
    .join(pd.get_dummies(new_pumpkins["City"]))
    .join(pd.get_dummies(new_pumpkins["Package"]))
)

y = new_pumpkins["Price"]

# then we just train based on these new data frame as before

# separate the train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# setup and train pipeline
# with max power 2
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:3.3} ({rmse / np.mean(y_pred) * 100:3.3}%)")

score = pipeline.score(X_train, y_train)
print("Model determination: ", score)
