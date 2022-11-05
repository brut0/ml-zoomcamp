""" Endpoint for churn prediction """

import json

import joblib
import numpy as np
import pandas as pd
from flask import Blueprint, Flask, Response, request

TARGET = "Churn"

CAT_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "SeniorCitizen",
]

NUM_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "TotalCharges_tenure"]

bp = Blueprint("ml", __name__)


@bp.route("/predict", methods=["POST"])
def predict_endpoint():
    status = 500
    result = None

    # TODO: Check input values
    try:
        data = request.get_json()
        df = pd.DataFrame(data, index=[0])
        df.tenure = df.tenure.astype(float)
        df.MonthlyCharges = df.MonthlyCharges.astype(float)

        total_ratio = joblib.load("TotalCharges_ratio.gz")
        df.TotalCharges = np.where(
            (df.TotalCharges == " ") | (df.TotalCharges == 0.0),
            df.MonthlyCharges * total_ratio,
            df.TotalCharges.astype(float),
        )
        df["TotalCharges_tenure"] = df.tenure / df.TotalCharges

        transformer = joblib.load("log1p.gz")
        df["TotalCharges"] = transformer.transform(df["TotalCharges"])

        scaler = joblib.load("scaler.gz")
        df[NUM_FEATURES] = scaler.transform(df[NUM_FEATURES])

        encoder = joblib.load("encoder.gz")
        test_ohe = encoder.transform(df[CAT_FEATURES])
        df[encoder.get_feature_names_out()] = test_ohe
        ALL_FEATURES_FINAL = [*NUM_FEATURES, *encoder.get_feature_names_out()]

        model = joblib.load("model.gz")
        predictions = model.predict(df[ALL_FEATURES_FINAL])
        df[TARGET] = ["Yes" if p else "No" for p in predictions]

        status = 200
        result = df[["customerID", TARGET]].set_index("customerID").to_dict()
        print(result)  # TODO: Add logger
    except Exception as e:
        print("Failed to predict")
        print(e)

    return Response(json.dumps(result), status=status, mimetype="application/json")


def create_app():
    flask_app = Flask("churn-prediction")
    flask_app.register_blueprint(bp)
    return flask_app


app = create_app()


if __name__ == "__main__":
    app.run()
