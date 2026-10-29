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
