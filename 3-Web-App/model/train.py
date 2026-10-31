import pandas as pd
import numpy as np

# NOTE:
# Run this script on the 3-Web-App/ dir

ufos = pd.read_csv("../data/ufos.csv")

ufos: pd.DataFrame = pd.DataFrame(
    {
        "Seconds": ufos["duration (seconds)"],
        "Country": ufos["country"],
        "Longitude": ufos["longitude"],
        "Latitude": ufos["latitude"],
    }
)

print(ufos.head())

print(ufos["Country"].unique())

# drop the invalid data
ufos.dropna(inplace=True)
ufos.info()
ufos = ufos[(ufos["Seconds"] >= 1) & (ufos["Seconds"] <= 60)]
ufos.info()

# encode the country (labels)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

label_encoder = LabelEncoder()
ufos["Country"] = label_encoder.fit_transform(ufos["Country"])


# print(ufos.head())

# features are all fields except Country
X = ufos[ufos.columns.difference(["Country"])]
y = ufos["Country"]


# for train converge faster, use the scaler
# scaler = StandardScaler()
# scaler.set_output(transform="pandas")
# X = scaler.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# as our output are category, so we will use LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred: np.ndarray = model.predict(X_test)

print(f"acc: {accuracy_score(y_test, y_pred)}")

# then save this model to file
import pickle

model_filename = "ufo-model.pkl"
# the wb means write as binary data
with open(model_filename, "wb") as dst:
    pickle.dump(model, dst)

# then this model has been saved into file
# we can load it like this
# with open(model_filename, "rb") as src:
#     new_model = pickle.load(src)
#     load_pred = new_model.predict(X_test)
#
#     diff: np.ndarray = y_pred - load_pred
#     diff = abs(diff)
#     # the output will be 0 for same model
#     print(f"difference predict output: {diff.sum()}")

# save the encoder
encoder_filename = "encoder.pkl"
with open(encoder_filename, "wb") as dst:
    pickle.dump(label_encoder, dst)
