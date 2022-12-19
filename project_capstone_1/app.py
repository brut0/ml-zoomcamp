""" Endpoint for churn prediction """

import json

import joblib
import numpy as np
import pandas as pd
from flask import Blueprint, Flask, Response, request

TARGET = "Battery Temperature [°C]"

bp = Blueprint("ml", __name__)


def preprocess_data(data: pd.DataFrame):
    df = data.copy()
    columns_join = [
        'Temperature Vent right [°C]',
        'Temperature Vent central right [°C]',
        'Temperature Vent central left [°C]',
        'Temperature Defrost lateral left [°C]',
        'Temperature Defrost lateral right [°C]',
        'Temperature Defrost central right [°C]',
        'Temperature Defrost central left [°C]'
    ]

    df['Temperature Vent Defrost'] = df[columns_join].apply(lambda x: max(x), axis=1)
    df = df.drop(columns_join, axis=1)

    return df


@bp.route("/predict", methods=["POST"])
def predict_endpoint():
    status = 500
    result = None

    # TODO: Check input values
    try:
        data = request.get_json()
        df = pd.DataFrame(data, index=[0])
        df = preprocess_data(data=df)

        scaler = joblib.load("scaler.gz")
        df = scaler.transform(df)

        model = joblib.load("model.gz")
        predictions = model.predict(df)

        status = 200
        result = {'predictions': [float(p) for p in predictions]}
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
