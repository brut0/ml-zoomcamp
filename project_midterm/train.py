import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils import resample

SEED = 42
N_FOLDS = 5

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

MODEL = RandomForestClassifier()
HYPER_PARAMS = {
    "max_depth": 5,
    "min_samples_leaf": 3,
    "min_samples_split": 3,
    "n_estimators": 100,
    "n_jobs": -1,
    "random_state": SEED,
}


def read_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def preprocess_data(data: pd.DataFrame):
    df = data.copy()
    df.loc[df.TotalCharges == " ", "TotalCharges"] = 0
    df.TotalCharges = df.TotalCharges.astype(float)

    df[TARGET] = df[TARGET].apply(lambda x: 1 if x == "Yes" else 0)

    df_majority = df[df[TARGET] == 0]
    df_minority = df[df[TARGET] == 1]

    df_minority_upsampled = resample(
        df_minority, replace=True, n_samples=df_minority.shape[0] * 2, random_state=SEED
    )

    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    X = df_upsampled.drop(columns=TARGET)
    y = df_upsampled[TARGET]

    mean_monthly = X.MonthlyCharges.mean()
    mean_total = X[X.TotalCharges != " "].TotalCharges.astype(float).mean()
    total_ratio = mean_total / mean_monthly
    joblib.dump(total_ratio, "TotalCharges_ratio.gz")

    X.TotalCharges = np.where(
        (X.TotalCharges == " ") | (X.TotalCharges == 0.0),
        X.MonthlyCharges * total_ratio,
        X.TotalCharges,
    )

    X["TotalCharges_tenure"] = X.tenure / X.TotalCharges

    return X, y


def transform_data(X):
    transformer = FunctionTransformer(np.log1p)
    X["TotalCharges"] = transformer.fit_transform(X["TotalCharges"])
    joblib.dump(transformer, "log1p.gz")
    print(f"FunctionTransformer saved to 'log1p.gz'")

    scaler = StandardScaler()
    X[NUM_FEATURES] = scaler.fit_transform(X[NUM_FEATURES])
    joblib.dump(scaler, "scaler.gz")
    print(f"StandardScaler saved to 'scaler.gz'")

    encoder = OneHotEncoder(handle_unknown="error", drop="if_binary", sparse=False)
    train_ohe = encoder.fit_transform(X[CAT_FEATURES])
    joblib.dump(encoder, "encoder.gz")
    print(f"OneHotEncoder of categorical features saved to 'encoder.gz'")

    X[encoder.get_feature_names_out()] = train_ohe
    X = X[[*NUM_FEATURES, *encoder.get_feature_names_out()]]

    return X


def main(data_path="WA_Fn-UseC_-Telco-Customer-Churn.csv"):
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
    print(classification_report(y, predictions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", metavar="path", required=True, help="the path to data"
    )
    args = parser.parse_args()
    main(data_path=args.data)
