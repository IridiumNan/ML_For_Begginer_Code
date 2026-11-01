import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from imblearn.over_sampling import SMOTE

pic_counter = 1


def ax_save_pic(ax):
    save_pic(ax.get_figure())


def save_pic(fig):
    global pic_counter
    save_path = f"./output/{pic_counter}.png"
    fig.savefig(save_path)
    pic_counter += 1


df: pd.DataFrame = pd.read_csv("../data/cuisines.csv")

print(df.head())

df.info()

print(f"all columns: {df.columns}")

ax = df["cuisine"].value_counts().plot.barh()
save_pic(ax.get_figure())

# separate data by different country
thai_df = df[(df["cuisine"] == "thai")]
japanese_df = df[(df["cuisine"] == "japanese")]
chinese_df = df[(df["cuisine"] == "chinese")]
indian_df = df[(df["cuisine"] == "indian")]
korean_df = df[(df["cuisine"] == "korean")]


print(f"thai df: {thai_df.shape}")
print(f"japanese df: {japanese_df.shape}")
print(f"chinese df: {chinese_df.shape}")
print(f"indian df: {indian_df.shape}")
print(f"korean df: {korean_df.shape}")


def create_ingredient_df(df: pd.DataFrame):
    ingredient_df = df.T.drop(["cuisine", "Unnamed: 0"]).sum(axis=1).to_frame("value")
    ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
    ingredient_df = ingredient_df.sort_values(by="value", ascending=False)
    return ingredient_df


thai_ingredient_df = create_ingredient_df(thai_df)
ax = thai_ingredient_df.head(10).plot.barh()
ax_save_pic(ax)

japanese_ingredient_df = create_ingredient_df(japanese_df)
ax = japanese_ingredient_df.head(10).plot.barh()
ax_save_pic(ax)

chinese_ingredient_df = create_ingredient_df(chinese_df)
ax = chinese_ingredient_df.head(10).plot.barh()
ax_save_pic(ax)

indian_ingredient_df = create_ingredient_df(indian_df)
ax = indian_ingredient_df.head(10).plot.barh()
ax_save_pic(ax)

korean_ingredient_df = create_ingredient_df(korean_df)
ax = korean_ingredient_df.head(10).plot.barh()
ax_save_pic(ax)

# drop the most common features
feature_df = df.drop(["cuisine", "Unnamed: 0", "rice", "garlic", "ginger"], axis=1)
label_df = df["cuisine"]
print(feature_df.head())


# generate new samples by over-sampling technique
oversample = SMOTE()
transformed_feature_fd, transformed_label_df = oversample.fit_resample(
    feature_df, label_df
)

print(f"new label count: {transformed_label_df.value_counts()}")
print(f"old label count: {label_df.value_counts()}")
