import argparse

import joblib, chardet
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

SEED = 42
N_FOLDS = 5

TARGET = "Battery Temperature [°C]"

MODEL = RandomForestRegressor()
HYPER_PARAMS = {
    "max_depth": 6,
    "min_samples_leaf": 2,
    "min_samples_split": 3,
    "n_estimators": 100,
    "n_jobs": -1,
    "random_state": SEED,
}


def read_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
            path,
            sep=";",
            encoding=chardet.detect(open(path, 'rb').read())['encoding']
        )
    return df


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

    columns_drop = [
        'Time [s]',
        'min. SoC [%]',
        'max. SoC [%)',
        'Heater Signal',
        'Requested Coolant Temperature [°C]',
        'Coolant Volume Flow +500 [l/h]',
        'Temperature Vent right [°C] ',
        'max. Battery Temperature [°C]',
        'SoC [%]',
        'Temperature Feetvent Driver [°C]',
        'displayed SoC [%]',
        'Heating Power LIN [W]',
        'Heating Power CAN [kW]',
        'Requested Heating Power [W]',
        'Coolant Volume Flow +500 [l/h]',
        'Longitudinal Acceleration [m/s^2]',
        'Temperature Coolant Heater Outlet [°C]',
        'Temperature Coolant Heater Inlet [°C]',
        'Heater Voltage [V]',
        'SoC [%]',
        'Temperature Heat Exchanger Outlet [°C]',
        'Heat Exchanger Temperature [°C]',
        'Coolant Temperature Heatercore [°C]'
    ]

    df['Temperature Vent Defrost'] = df[columns_join].apply(lambda x: max(x), axis=1)
    df = df.drop(columns_join, axis=1)
    df = df.drop(columns_drop, axis=1)
    
    X = df.drop(columns=TARGET)
    y = df[TARGET]

    return X, y


def transform_data(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.gz")
    print(f"StandardScaler saved to 'scaler.gz'")

    return X


def main(data_path):
    df = read_data(data_path)
    X, y = preprocess_data(df)
    X = transform_data(X)

    model = MODEL
    model.set_params(**HYPER_PARAMS)
    model.fit(X, y)
    joblib.dump(model, "model.gz")
    print(f"Fitted model saved to 'model.gz'")

    predictions = model.predict(X)
    print("Classification metrics:")
    print(mean_squared_error(y, predictions, squared=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", metavar="path", required=True, help="the path to data"
    )
    args = parser.parse_args()
    main(data_path=args.data)
