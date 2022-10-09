from flask import Flask, request, jsonify
import pickle


def load_preprocessor(fname):
    with open(fname, "rb") as file:
        dv = pickle.load(file)
    return dv


def load_model(fname):
    with open(fname, "rb") as file:
        model = pickle.load(file)
    return model


def predict_one(dv, model, data):
    X = dv.transform([data])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


dv = load_preprocessor("dv.bin")
model = load_model("model1.bin")

app = Flask("app")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    pred = predict_one(dv, model, data)
    result = {"prob": float(pred)}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
