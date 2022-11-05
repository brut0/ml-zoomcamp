# ML Zoomcamp Midterm Project

## Problem definition: Telco customer churn
The main goal is to use relevant customer data to predict behavior to retain customers. The raw data contains 7043 rows (customers) and 21 columns (features).
Dataset source: [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)


## Repository structure
- **EDA.ipynb**: Jupyter notebooks with EDA and experiments with models
- **train.py**: Automated scripts to train model and save artifacts localy
- **app.py**: Prediction endpoint using Flask
- **test_app.py**: Script to test endpoint

## Train model

    virtualenv env
    source env/bin/activate
    pip install -r requirements.txt
    python3 train.py --data=<data_path>


## Run Flask app

    export FLASK_APP=app.py
    export FLASK_RUN_PORT=8800
    flask run

Test endpoint:

    export FLASK_RUN_PORT=8800
    python3 test_app.py
