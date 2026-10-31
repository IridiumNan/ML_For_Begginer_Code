import pickle
import socket
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

with open("./ufo-model.pkl", "rb") as src:
    model = pickle.load(src)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]
    return render_template(
        "index.html", predict_text=f"Likely country: {countries[output]}"
    )


# TODO: Fix the css bug
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8081)
