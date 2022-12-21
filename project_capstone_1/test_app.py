import argparse
import os, chardet

import pandas as pd
import requests

FEATURES = [
    'Temperature Vent right [°C]',
    'Temperature Vent central right [°C]',
    'Temperature Vent central left [°C]',
    'Temperature Defrost lateral left [°C]',
    'Temperature Defrost lateral right [°C]',
    'Temperature Defrost central right [°C]',
    'Temperature Defrost central left [°C]',
    'Throttle [%]', 'Regenerative Braking Signal ',
    'Coolant Temperature Inlet [°C]', 'Battery Voltage [V]',
    'Elevation [m]', 'Cabin Temperature Sensor [°C]', 'Battery Current [A]',
    'Velocity [km/h]', 'Temperature Head Co-Driver [°C]',
    'Ambient Temperature Sensor [°C]',
    'Temperature Feetvent Co-Driver [°C]',
    'Temperature Defrost central [°C]', 'Ambient Temperature [°C]',
    'Temperature Footweel Driver [°C]', 'AirCon Power [kW]',
    'Temperature Head Driver [°C]', 'Heater Current [A]',
    'Temperature Footweel Co-Driver [°C]', 'Motor Torque [Nm]',
]


def read_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
            path,
            sep=";",
            encoding=chardet.detect(open(path, 'rb').read())['encoding']
        )
    return df


def main(data_path):
    df = read_data(data_path)
    data = df[FEATURES].iloc[0].to_dict()
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
