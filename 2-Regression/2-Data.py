import pandas as pd
import os

# NOTE:
# RUn this script on 2-Regression

pumpkins = pd.read_csv("../data/US-pumpkins.csv")
# shape
# (1757, 26)

# just get all packages contains "bushel"
# (415, 26)
pumpkins = pumpkins[pumpkins["Package"].str.contains("bushel", case=True, regex=True)]


print(pumpkins.head())
# check nan values
print(pumpkins.isnull().sum())

# select specific columns
column_to_select = ["Package", "Low Price", "High Price", "Date"]
# first dim: use : for all samples
# second dim: selected columns
pumpkins = pumpkins.loc[:, column_to_select]

price = (pumpkins["Low Price"] + pumpkins["High Price"]) / 2
month = pd.DatetimeIndex(pumpkins["Date"]).month

new_pumpkins = pd.DataFrame(
    {
        "Month": month,
        "Package": pumpkins["Package"],
        "Low Price": pumpkins["Low Price"],
        "High Price": pumpkins["High Price"],
        "Price": price,
    }
)

new_pumpkins.loc[new_pumpkins["Package"].str.contains("1 1/9"), "Price"] = price / (
    1 + 1 / 9
)
new_pumpkins.loc[new_pumpkins["Package"].str.contains("1/2"), "Price"] = price / (1 / 2)

# now we have preprocess the data
#
# then we can visualize them

import matplotlib.pyplot as plt

price = new_pumpkins["Price"]
month = new_pumpkins["Month"]
plt.xlabel("month")
plt.ylabel("price")

plt.scatter(month, price)
plt.savefig("./output/2.png")

# draw a more useful picture
month_price = new_pumpkins.groupby(["Month"])["Price"].mean()
month_price.plot(kind="bar")

plt.ylabel("Pumpkin Price")
plt.ylim(ymin=0, ymax=month_price.max() + 10)
plt.savefig("./output/3.png")


# draw a graph by using seaborn lib

import seaborn as sns

fg: sns.FacetGrid = sns.relplot(x="Month", y="Price", data=new_pumpkins)
fg.savefig("./output/4.png")

fg: sns.FacetGrid = sns.catplot(x="Month", y="Price", data=new_pumpkins, kind="bar")
fg.savefig("./output/5.png")

# draw a heatmap to show relationship between every pair of columns

correlations = new_pumpkins[["Month", "Low Price", "High Price", "Price"]].corr()
# plt.Axes
ax = sns.heatmap(data=correlations, annot=True, cmap="coolwarm")
# plt.Figure
fg = ax.get_figure()
fg.savefig("./output/6.png")
