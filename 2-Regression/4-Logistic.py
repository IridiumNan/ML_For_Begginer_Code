import numpy as np
import pandas as pd
import seaborn as sns

full_pumpkins = pd.read_csv("../data/US-pumpkins.csv")

column_to_select = [
    "City Name",
    "Package",
    "Variety",
    "Origin",
    "Item Size",
    "Color",
]

# select columns and drop all null value
pumpkins: pd.DataFrame = full_pumpkins.loc[:, column_to_select]
pumpkins.dropna(inplace=True)

pumpkins.info()

# categorical plot
# using seaborn

palette = {"ORANGE": "orange", "WHITE": "wheat"}
fg = sns.catplot(data=pumpkins, y="Variety", hue="Color", kind="count", palette=palette)
fg.savefig("./output/12.png")

# label encoding
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder

item_size_categories = [
    [
        "sml",
        "med",
        "med-lge",
        "lge",
        "xlge",
        "jbo",
        "exjbo",
    ]
]

# the ordinal encoder
# apply this encoder when these categories has logical order
# encode the category into number
ordinal_features = ["Item Size"]
ordinal_encoder = OrdinalEncoder(categories=item_size_categories)

categorical_features = ["City Name", "Package", "Variety", "Origin"]
categorical_encoder = OneHotEncoder(sparse_output=False)

from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    transformers=[
        ("ord", ordinal_encoder, ordinal_features),
        ("cat", categorical_encoder, categorical_features),
    ]
)

# NOTE:
# important, else fit_transform will return the numpy.ndarray
ct.set_output(transform="pandas")
encoded_features: pd.DataFrame = ct.fit_transform(pumpkins)

label_encoder = LabelEncoder()
encoded_label = label_encoder.fit_transform(pumpkins["Color"])


encoded_pumpkins: pd.DataFrame = encoded_features.assign(Color=encoded_label)

pumpkins["Item Size"] = encoded_pumpkins["ord__Item Size"]

g = sns.catplot(
    data=pumpkins,
    x="Item Size",
    y="Color",
    row="Variety",
    kind="box",
    orient="h",
    sharex=False,
    margin_titles=True,
    height=1.8,
    aspect=4,
    palette=palette,
)
g.savefig("./output/13.png")


# training a model
# using sigmoid function

from sklearn.model_selection import train_test_split

# X contains all features except Color
X = encoded_pumpkins[encoded_pumpkins.columns.difference(["Color"])]
y = encoded_pumpkins["Color"]

# then split into train set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression

# create then train this logistic model
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print("Predicted labels: ", y_pred)
print("F1-score: ", f1_score(y_test, y_pred))

# get a confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm: np.ndarray = confusion_matrix(y_test, y_pred)

print(f"confusion matrix: {cm}")

# config the label as encoder classes
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=label_encoder.classes_
)

disp.plot()

import matplotlib.pyplot as plt

plt.savefig("./output/14.png")

# draw a ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score

y_scores = model.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])

fig = plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], "k--")
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("./output/15.png")

auc = roc_auc_score(y_test, y_scores[:, 1])
print(f"auc: {auc}")
