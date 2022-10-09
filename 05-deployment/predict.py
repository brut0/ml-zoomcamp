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


if __name__ == "__main__":
    client_data = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}
    dv = load_preprocessor("dv.bin")
    model = load_model("model1.bin")

    y_pred = predict_one(dv, model, client_data)
    print(y_pred)
