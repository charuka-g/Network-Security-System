from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "charuka",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def run_training_pipeline():
    from src.pipeline import TrainingPipeline

    pipeline = TrainingPipeline()
    artifact = pipeline.run_pipeline()
    print(f"Training complete — Test F1: {artifact.test_metric.f1_score:.4f}")


with DAG(
    dag_id="network_security_training",
    default_args=default_args,
    description="End-to-end training pipeline for the phishing detection model",
    schedule_interval="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["network-security", "mlops"],
) as dag:
    train_task = PythonOperator(
        task_id="run_training_pipeline",
        python_callable=run_training_pipeline,
    )
