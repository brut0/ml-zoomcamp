# ML Zoomcamp Midterm Project

## Problem definition: Telco customer churn
The main goal is to use relevant customer data to predict behavior to retain customers. The raw data contains 7043 rows (customers) and 21 columns (features).
Dataset source: [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)


## Repository structure
- **EDA.ipynb**: Jupyter notebooks with EDA and experiments with models
- **train.py**: Automated scripts to train model and save artifacts localy
- **app.py**: Prediction endpoint using Flask
- **test_app.py**: Script to test endpoint
- **Dockerfile**: Docker file
- **docker_entrypoint.sh**: Entrypoint in docker to run app
- **docker-compose.yml**: Docker Compose file


## Train model
Activate environment, install dependecies and run train pipeline

    virtualenv env
    source env/bin/activate
    pip install -r requirements.txt
    python3 train.py --data=<data_path>

_Don't forget to download dataset before!_

## Run Flask app

    export FLASK_APP=app.py
    export FLASK_RUN_PORT=8800
    flask run

Test endpoint:

    export FLASK_RUN_PORT=8800
    python3 test_app.py

Response format of endpoint:

    {
        'Churn':
            {<ID>: <prediction_yes_or_not>}
    }


## Dockerize
Run just one simple command (be sure that train model before)

    docker compose up

