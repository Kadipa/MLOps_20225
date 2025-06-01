from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import logging
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
import os


def train_and_log_model():
    # Set MLflow tracking URI (ensure volume is mounted to /opt/airflow/mlruns)
    mlflow.set_tracking_uri("file:///opt/airflow/mlruns")
    mlflow.set_experiment("nyc-taxi-experiment")

    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
    df = pd.read_parquet(url)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df['duration']

    model = LinearRegression()
    model.fit(X_train, y_train)

    logging.info(f"Model intercept: {model.intercept_:.2f}")

    with mlflow.start_run():
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_dict(dv.get_feature_names_out().tolist(), "dv_features.json")


default_args = {
    "owner": "airflow",
    "start_date": datetime(2023, 1, 1),
}

dag = DAG(
    dag_id="nyc_taxi_hw3_dag",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["homework"],
)

train_task = PythonOperator(
    task_id="train_and_log_model",
    python_callable=train_and_log_model,
    dag=dag,
)
