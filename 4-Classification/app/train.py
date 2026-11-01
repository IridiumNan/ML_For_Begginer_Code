import pandas as pd


data = pd.read_csv("../../data/cleaned_cuisine.csv")

X = data.drop(["Unnamed: 0", "cuisine"], axis=1, inplace=False)
y = data["cuisine"]

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel="linear", C=10, probability=True, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"the accuracy: {acc}")
print(classification_report(y_test, y_pred))

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [("float_input", FloatTensorType([None, 380]))]
options = {id(model): {"nocl": True, "zipmap": False}}

onx = convert_sklearn(model, initial_types=initial_type, options=options)

save_path = "./model.onnx"
try:
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    print("model has been saved to ", save_path)
except OSError:
    print("error while opening file")
