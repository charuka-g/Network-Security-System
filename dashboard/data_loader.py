"""Data loading and metrics computation for the dashboard."""

import os
import glob
import pickle
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "Artifacts")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PREDICTION_DIR = os.path.join(BASE_DIR, "prediction_output")

FEATURE_CATEGORIES = {
    "URL-Based": [
        "having_IP_Address", "URL_Length", "Shortining_Service",
        "having_At_Symbol", "double_slash_redirecting", "Prefix_Suffix",
        "having_Sub_Domain", "HTTPS_token",
    ],
    "Domain-Based": [
        "SSLfinal_State", "Domain_registeration_length", "Favicon",
        "port", "Request_URL",
    ],
    "Page-Based": [
        "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email",
        "Abnormal_URL", "Redirect", "on_mouseover", "RightClick",
        "popUpWidnow", "Iframe",
    ],
    "Reputation-Based": [
        "age_of_domain", "DNSRecord", "web_traffic", "Page_Rank",
        "Google_Index", "Links_pointing_to_page", "Statistical_report",
    ],
}


def get_latest_artifact_dir() -> str | None:
    """Return the most recent artifact directory path."""
    if not os.path.isdir(ARTIFACTS_DIR):
        return None
    dirs = sorted(glob.glob(os.path.join(ARTIFACTS_DIR, "*")), key=os.path.getmtime)
    return dirs[-1] if dirs else None


def load_raw_data() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "phisingData.csv")
    return pd.read_csv(path)


def load_train_test(artifact_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(artifact_dir, "data_validation", "train.csv")
    test_path = os.path.join(artifact_dir, "data_validation", "test.csv")
    if not os.path.exists(train_path):
        train_path = os.path.join(artifact_dir, "data_ingestion", "train.csv")
        test_path = os.path.join(artifact_dir, "data_ingestion", "test.csv")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def load_model():
    path = os.path.join(MODELS_DIR, "model.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_preprocessor():
    path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_drift_report(artifact_dir: str) -> dict | None:
    path = os.path.join(artifact_dir, "data_validation", "drift_report.yaml")
    if not os.path.exists(path):
        # Try older directory structure
        path = os.path.join(artifact_dir, "data_validation", "drift_report", "report.yaml")
        if not os.path.exists(path):
            return None
    try:
        with open(path, "r") as f:
            return yaml.full_load(f)
    except Exception:
        # Fallback: parse p_values manually from the YAML text
        report = {}
        with open(path, "r") as f:
            content = f.read()
        current_feature = None
        for line in content.split("\n"):
            stripped = line.strip()
            if not stripped.startswith("-") and stripped.endswith(":") and not stripped.startswith("drift") and not stripped.startswith("p_value"):
                current_feature = stripped[:-1]
            elif stripped.startswith("p_value:") and current_feature:
                try:
                    p_val = float(stripped.split(":")[1].strip())
                    report[current_feature] = {"p_value": p_val, "drift": p_val < 0.05}
                except ValueError:
                    pass
        return report if report else None


def load_prediction_output() -> pd.DataFrame | None:
    path = os.path.join(PREDICTION_DIR, "output.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _normalize_labels(y_true, y_pred):
    """Map labels to 0 (legitimate) and 1 (phishing) for consistent metrics.

    Handles the case where actuals are {-1, 1} but model predicts {0, 1}
    (due to KNN imputer transforming the target during training).
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    # If actuals contain -1 but predictions don't, map -1 -> 0
    if -1 in y_true and -1 not in y_pred:
        y_true = np.where(y_true == -1, 0, y_true)
    return y_true.astype(int), y_pred.astype(int)


def compute_full_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Compute comprehensive classification metrics."""
    y_true, y_pred = _normalize_labels(y_true, y_pred)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, pos_label=1),
        "precision": precision_score(y_true, y_pred, pos_label=1),
        "recall": recall_score(y_true, y_pred, pos_label=1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=["Legitimate", "Phishing"], output_dict=True,
        ),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
        metrics["roc_curve"] = (fpr, tpr)
        prec, rec, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
        metrics["pr_curve"] = (prec, rec)
    return metrics


def get_feature_importance(model, feature_names: list[str]) -> pd.DataFrame | None:
    """Extract feature importance from the model."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_).flatten()
    else:
        return None
    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def get_all_artifact_runs() -> list[dict]:
    """List all artifact runs with their timestamps."""
    if not os.path.isdir(ARTIFACTS_DIR):
        return []
    runs = []
    for d in sorted(glob.glob(os.path.join(ARTIFACTS_DIR, "*")), key=os.path.getmtime, reverse=True):
        name = os.path.basename(d)
        has_model = os.path.exists(os.path.join(d, "model_trainer", "model.pkl"))
        runs.append({"name": name, "path": d, "has_model": has_model})
    return runs
