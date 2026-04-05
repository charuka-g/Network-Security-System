import os
import dagshub
import mlflow

_initialized = False


def init_dagshub() -> None:
    """Initialise DagsHub + MLflow tracking (idempotent)."""
    global _initialized
    if _initialized:
        return

    repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "charukagunawardhaneixvii")
    repo_name = os.getenv("DAGSHUB_REPO_NAME", "Network-Security-System")
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    _initialized = True
