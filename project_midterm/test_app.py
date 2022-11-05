import argparse
import os

import pandas as pd
import requests


def main(data_path="WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    df = pd.read_csv(data_path)
    data = df.iloc[0].to_dict()
    url = f"http://localhost:{os.environ['FLASK_RUN_PORT']}/predict"
    response = requests.post(url, json=data, timeout=3)
    print(response)
    print(response.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", metavar="path", required=False, help="the path to data"
    )
    args = parser.parse_args()
    if args.data:
        main(data_path=args.data)
    else:
        main()
