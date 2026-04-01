"""
Network Security — Phishing Detection Dashboard
Enterprise-grade monitoring and analytics dashboard.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from dashboard.data_loader import (
    FEATURE_CATEGORIES,
    _normalize_labels,
    compute_full_metrics,
    get_all_artifact_runs,
    get_feature_importance,
    get_latest_artifact_dir,
    load_drift_report,
    load_model,
    load_prediction_output,
    load_preprocessor,
    load_raw_data,
    load_train_test,
)


def _label_name(val):
    """Map numeric label to human-readable name, handling both -1/1 and 0/1."""
    if val in (1, 1.0):
        return "Phishing"
    return "Legitimate"

# ──────────────────────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Network Security | Phishing Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Enterprise palette + typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
    :root {
        --bg: #0b1220;
        --sidebar: #0a1020;
        --surface: rgba(255, 255, 255, 0.04);
        --surface-2: rgba(255, 255, 255, 0.06);
        --border: rgba(148, 163, 184, 0.18);
        --text: #e7eefc;
        --muted: rgba(231, 238, 252, 0.72);
        --muted-2: rgba(231, 238, 252, 0.56);
        --accent: #4f86ff;
        --accent-2: #7aa2ff;
        --danger: #ff5c77;
        --success: #34d399;
        --warning: #fbbf24;
        --shadow: 0 10px 30px rgba(0,0,0,0.35);
        --radius: 14px;
        --radius-sm: 10px;
    }

    /* App background */
    .stApp {
        background: radial-gradient(1200px 800px at 15% 10%, rgba(79,134,255,0.12) 0%, rgba(11,18,32,0.0) 55%),
                    radial-gradient(900px 600px at 90% 15%, rgba(122,162,255,0.08) 0%, rgba(11,18,32,0.0) 55%),
                    var(--bg);
        color: var(--text);
        font-family: "IBM Plex Sans", "Inter", ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--sidebar);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: var(--muted);
    }

    /* Headings */
    h1, h2, h3, h4 { color: var(--text) !important; }
    h1 { font-weight: 750 !important; letter-spacing: -0.02em; }

    /* Links */
    a { color: var(--accent-2) !important; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, var(--surface-2), var(--surface));
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 14px 16px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(10px);
    }
    div[data-testid="stMetric"] label {
        color: var(--muted-2) !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-size: 1.9rem !important;
        font-weight: 700 !important;
        font-family: "Inter", ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: 1px solid var(--border);
        border-bottom: 0;
        border-radius: 12px 12px 0 0;
        color: var(--muted-2);
        padding: 10px 18px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, var(--surface-2), rgba(255,255,255,0.03)) !important;
        color: var(--text) !important;
        border-color: rgba(79,134,255,0.35) !important;
        box-shadow: 0 8px 18px rgba(0,0,0,0.25);
    }

    /* Inputs & controls */
    .stTextInput > div > div,
    .stNumberInput > div > div,
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stDateInput > div > div {
        background: rgba(255,255,255,0.035) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
    }
    .stRadio > div, .stCheckbox > div { color: var(--muted) !important; }

    /* Buttons */
    button[kind="primary"] {
        background: linear-gradient(180deg, rgba(79,134,255,1), rgba(79,134,255,0.85)) !important;
        border: 1px solid rgba(79,134,255,0.55) !important;
        border-radius: 12px !important;
        box-shadow: 0 12px 28px rgba(79,134,255,0.18);
        font-weight: 650 !important;
    }
    button[kind="secondary"] {
        background: rgba(255,255,255,0.035) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        color: var(--text) !important;
    }

    /* Plotly chart containers */
    .stPlotlyChart {
        background: linear-gradient(180deg, var(--surface-2), var(--surface));
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 10px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(10px);
    }

    /* Dataframes */
    .stDataFrame {
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        overflow: hidden;
    }

    /* Alerts */
    .stAlert {
        border-radius: var(--radius-sm);
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.03);
    }

    /* Divider */
    hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Plotly theme
# ──────────────────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, Inter, sans-serif", color="#e7eefc"),
    margin=dict(l=40, r=20, t=40, b=40),
)

COLORS = {
    "primary": "#4f86ff",
    "success": "#34d399",
    "danger": "#ff5c77",
    "warning": "#fbbf24",
    "info": "#7aa2ff",
    "neutral": "rgba(231, 238, 252, 0.65)",
    "neutral_weak": "rgba(231, 238, 252, 0.35)",
    "accent_scale": ["#2f65ff", "#4f86ff", "#7aa2ff"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Network Security")
    st.markdown("**Phishing Detection Dashboard**")
    st.markdown("---")

    runs = get_all_artifact_runs()
    if runs:
        run_names = [r["name"] for r in runs]
        selected_run = st.selectbox(
            "Select training run",
            run_names,
            index=0,
            help="Choose an artifact run to analyze",
        )
        artifact_dir = next(r["path"] for r in runs if r["name"] == selected_run)
    else:
        st.warning("No artifact runs found.")
        artifact_dir = None

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "Go to",
        [
            "Executive Summary",
            "Model Performance",
            "Feature Analysis",
            "Data Drift Monitor",
            "Prediction Explorer",
            "Live Prediction",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:rgba(231, 238, 252, 0.5); font-size:0.75rem;'>"
        "v1.0.0 · Streamlit<br>© 2026 Network Security Team"
        "</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def cached_load_raw_data():
    return load_raw_data()


@st.cache_data
def cached_load_train_test(artifact_dir):
    return load_train_test(artifact_dir)


@st.cache_resource
def cached_load_model():
    return load_model()


@st.cache_resource
def cached_load_preprocessor():
    return load_preprocessor()


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Executive Summary
# ──────────────────────────────────────────────────────────────────────────────
if page == "Executive Summary":
    st.markdown("# Executive Summary")
    st.markdown("High-level overview of the phishing detection system's performance and data health.")
    st.markdown("---")

    if artifact_dir is None:
        st.error("No training artifacts found. Please run the training pipeline first.")
        st.stop()

    raw_df = cached_load_raw_data()
    train_df, test_df = cached_load_train_test(artifact_dir)
    model = cached_load_model()
    preprocessor = cached_load_preprocessor()

    # KPI row
    col1, col2, col3, col4 = st.columns(4)

    target_col = "Result"
    features = [c for c in test_df.columns if c != target_col]

    if model is not None and preprocessor is not None:
        X_test = test_df[features]
        y_test = test_df[target_col]
        X_test_transformed = preprocessor.transform(X_test)
        y_pred = model.predict(X_test_transformed)

        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_transformed)[:, 1]

        metrics = compute_full_metrics(y_test, y_pred, y_prob)

        with col1:
            st.metric("F1 Score", f"{metrics['f1_score']:.3f}", help="Harmonic mean of precision and recall")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.3f}", help="True positives / predicted positives")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.3f}", help="True positives / actual positives")
        with col4:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}", help="Overall prediction accuracy")

        st.markdown("")

        # Second KPI row
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Total Samples", f"{len(raw_df):,}")
        with col6:
            st.metric("Training Set", f"{len(train_df):,}")
        with col7:
            st.metric("Test Set", f"{len(test_df):,}")
        with col8:
            phishing_rate = (raw_df[target_col] == 1).mean() * 100
            st.metric("Phishing Rate", f"{phishing_rate:.1f}%")

        st.markdown("---")

        # Charts row
        left, right = st.columns(2)

        with left:
            st.markdown("### Class Distribution")
            class_counts = raw_df[target_col].value_counts().reset_index()
            class_counts.columns = ["Class", "Count"]
            class_counts["Label"] = class_counts["Class"].map(_label_name)
            fig = px.pie(
                class_counts,
                values="Count",
                names="Label",
                color="Label",
                color_discrete_map={"Phishing": COLORS["danger"], "Legitimate": COLORS["success"], "Suspicious": COLORS["warning"]},
                hole=0.45,
            )
            fig.update_layout(**PLOT_LAYOUT, showlegend=True, height=350)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("### Model Comparison (Test F1)")
            model_name = type(model).__name__
            # Show metrics breakdown as bar chart
            metric_names = ["Accuracy", "F1 Score", "Precision", "Recall"]
            metric_vals = [metrics["accuracy"], metrics["f1_score"], metrics["precision"], metrics["recall"]]
            if "roc_auc" in metrics:
                metric_names.append("ROC AUC")
                metric_vals.append(metrics["roc_auc"])

            fig = go.Figure(go.Bar(
                x=metric_names,
                y=metric_vals,
                marker_color=[COLORS["primary"]] * len(metric_names),
                text=[f"{v:.3f}" for v in metric_vals],
                textposition="outside",
            ))
            fig.update_layout(**PLOT_LAYOUT, height=350, yaxis=dict(range=[0, 1.1]), title=f"Active Model: {model_name}")
            st.plotly_chart(fig, use_container_width=True)

        # Drift summary
        drift_report = load_drift_report(artifact_dir)
        if drift_report:
            st.markdown("### Data Drift Summary")
            drift_count = sum(1 for v in drift_report.values() if isinstance(v, dict) and v.get("p_value", 1) < 0.05)
            total_features = len(drift_report)
            if drift_count == 0:
                st.success(f"No data drift detected across all {total_features} features.")
            else:
                st.warning(f"Data drift detected in {drift_count}/{total_features} features.")
    else:
        st.error("Model or preprocessor not found. Please train the model first.")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Model Performance
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Model Performance":
    st.markdown("# Model Performance")
    st.markdown("Detailed classification metrics, confusion matrix, ROC and Precision-Recall curves.")
    st.markdown("---")

    if artifact_dir is None:
        st.error("No training artifacts found.")
        st.stop()

    train_df, test_df = cached_load_train_test(artifact_dir)
    model = cached_load_model()
    preprocessor = cached_load_preprocessor()

    if model is None or preprocessor is None:
        st.error("Model or preprocessor not found.")
        st.stop()

    target_col = "Result"
    features = [c for c in test_df.columns if c != target_col]

    dataset_choice = st.radio("Evaluate on:", ["Test Set", "Training Set"], horizontal=True)
    eval_df = test_df if dataset_choice == "Test Set" else train_df

    X_eval = eval_df[features]
    y_eval = eval_df[target_col]
    X_transformed = preprocessor.transform(X_eval)
    y_pred = model.predict(X_transformed)
    y_prob = model.predict_proba(X_transformed)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = compute_full_metrics(y_eval, y_pred, y_prob)

    # Metrics cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("F1 Score", f"{metrics['f1_score']:.4f}")
    c3.metric("Precision", f"{metrics['precision']:.4f}")
    c4.metric("Recall", f"{metrics['recall']:.4f}")
    c5.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.4f}" if y_prob is not None else "N/A")

    st.markdown("---")

    # Charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Confusion Matrix")
        cm = metrics["confusion_matrix"]
        labels = ["Legitimate", "Phishing"]
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}",
            textfont={"size": 20, "color": "white"},
            colorscale=[[0, "rgba(231, 238, 252, 0.10)"], [1, COLORS["primary"]]],
            showscale=False,
        ))
        fig.update_layout(
            **PLOT_LAYOUT,
            height=400,
            xaxis_title="Predicted",
            yaxis_title="Actual",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        if y_prob is not None:
            st.markdown("### ROC Curve")
            fpr, tpr = metrics["roc_curve"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f'ROC (AUC={metrics["roc_auc"]:.3f})',
                line=dict(color=COLORS["primary"], width=3),
                fill="tozeroy",
                fillcolor="rgba(79, 134, 255, 0.14)",
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="Random",
                line=dict(color=COLORS["neutral_weak"], width=1, dash="dash"),
            ))
            fig.update_layout(**PLOT_LAYOUT, height=400, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ROC curve not available — model does not support probability estimates.")

    # PR Curve
    if y_prob is not None:
        col_pr, col_report = st.columns(2)
        with col_pr:
            st.markdown("### Precision-Recall Curve")
            prec, rec = metrics["pr_curve"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rec, y=prec, mode="lines",
                name="PR Curve",
                line=dict(color=COLORS["info"], width=3),
                fill="tozeroy",
                fillcolor="rgba(122, 162, 255, 0.14)",
            ))
            fig.update_layout(**PLOT_LAYOUT, height=400, xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(fig, use_container_width=True)

        with col_report:
            st.markdown("### Classification Report")
            report = metrics["classification_report"]
            report_df = pd.DataFrame(report).transpose()
            report_df = report_df.round(4)
            st.dataframe(report_df, use_container_width=True, height=350)

    # Score distribution
    if y_prob is not None:
        st.markdown("### Prediction Score Distribution")
        score_df = pd.DataFrame({"Probability": y_prob, "Actual": y_eval.values})
        score_df["Label"] = score_df["Actual"].map(_label_name)
        fig = px.histogram(
            score_df, x="Probability", color="Label", nbins=50, barmode="overlay",
            color_discrete_map={"Phishing": COLORS["danger"], "Legitimate": COLORS["success"]},
            opacity=0.7,
        )
        fig.update_layout(**PLOT_LAYOUT, height=350, xaxis_title="Predicted Probability", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Feature Analysis
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Feature Analysis":
    st.markdown("# Feature Analysis")
    st.markdown("Feature importance, correlations, and distribution analysis.")
    st.markdown("---")

    model = cached_load_model()
    raw_df = cached_load_raw_data()
    target_col = "Result"
    features = [c for c in raw_df.columns if c != target_col]

    if model is not None:
        importance_df = get_feature_importance(model, features)
        if importance_df is not None:
            st.markdown("### Feature Importance (Top 15)")

            top_15 = importance_df.head(15)
            # Add category info
            cat_map = {}
            for cat, feats in FEATURE_CATEGORIES.items():
                for f in feats:
                    cat_map[f] = cat
            top_15 = top_15.copy()
            top_15["category"] = top_15["feature"].map(cat_map).fillna("Other")

            fig = px.bar(
                top_15.iloc[::-1],
                x="importance",
                y="feature",
                color="category",
                orientation="h",
                color_discrete_sequence=COLORS["accent_scale"],
            )
            fig.update_layout(**PLOT_LAYOUT, height=500, xaxis_title="Importance", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")

    # Correlation heatmap
    st.markdown("### Feature Correlation Matrix")

    corr_view = st.selectbox("View:", ["Top correlated with target", "Full matrix (sampled)"])

    if corr_view == "Top correlated with target":
        corr_with_target = raw_df[features + [target_col]].corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
        top_corr = corr_with_target.head(15)
        fig = go.Figure(go.Bar(
            x=top_corr.values,
            y=top_corr.index,
            orientation="h",
            marker_color=COLORS["info"],
            text=[f"{v:.3f}" for v in top_corr.values],
            textposition="outside",
        ))
        fig.update_layout(**PLOT_LAYOUT, height=450, xaxis_title="|Correlation| with Target", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        top_features = raw_df[features].corr()[target_col if target_col in raw_df[features].columns else features[0]].abs().sort_values(ascending=False).head(12).index.tolist() if target_col in raw_df.columns else features[:12]
        # Use top correlated features for readable heatmap
        corr_feats = raw_df[features + [target_col]].corr()[target_col].drop(target_col).abs().sort_values(ascending=False).head(12).index.tolist()
        corr_matrix = raw_df[corr_feats].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdBu_r",
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
        ))
        fig.update_layout(**PLOT_LAYOUT, height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Feature distributions by category
    st.markdown("### Feature Distributions by Category")
    selected_cat = st.selectbox("Category:", list(FEATURE_CATEGORIES.keys()))
    cat_features = FEATURE_CATEGORIES[selected_cat]

    fig = make_subplots(
        rows=2,
        cols=min(4, (len(cat_features) + 1) // 2),
        subplot_titles=cat_features[: 8],
    )
    for i, feat in enumerate(cat_features[:8]):
        row = i // 4 + 1
        col = i % 4 + 1
        for label, color in [(-1, COLORS["success"]), (1, COLORS["danger"])]:
            subset = raw_df[raw_df[target_col] == label][feat]
            fig.add_trace(
                go.Histogram(x=subset, name=f"{'Legit' if label == -1 else 'Phish'}", marker_color=color, opacity=0.6, showlegend=(i == 0)),
                row=row, col=col,
            )
    fig.update_layout(**PLOT_LAYOUT, height=400, barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Data Drift Monitor
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Data Drift Monitor":
    st.markdown("# Data Drift Monitor")
    st.markdown("Statistical drift detection between training and test distributions using the Kolmogorov-Smirnov test.")
    st.markdown("---")

    if artifact_dir is None:
        st.error("No training artifacts found.")
        st.stop()

    drift_report = load_drift_report(artifact_dir)

    if drift_report is None:
        st.warning("No drift report found for this run.")
        st.stop()

    # Parse drift data
    drift_data = []
    for feature, values in drift_report.items():
        if isinstance(values, dict) and "p_value" in values:
            p_val = values["p_value"]
            drifted = p_val < 0.05
            drift_data.append({
                "Feature": feature,
                "P-Value": p_val,
                "Drift Detected": drifted,
                "Status": "Drift" if drifted else "Stable",
                "Confidence": (1 - p_val) * 100,
            })

    drift_df = pd.DataFrame(drift_data).sort_values("P-Value")

    # Summary metrics
    total = len(drift_df)
    drifted = drift_df["Drift Detected"].sum()
    stable = total - drifted
    avg_p = drift_df["P-Value"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Features", total)
    c2.metric("Stable", stable, help="Features with p-value >= 0.05")
    c3.metric("Drifted", int(drifted), delta=f"-{int(drifted)}" if drifted > 0 else "0", delta_color="inverse")
    c4.metric("Avg P-Value", f"{avg_p:.4f}")

    st.markdown("---")

    # Drift heatmap
    st.markdown("### Drift P-Values by Feature")
    drift_sorted = drift_df.sort_values("P-Value")
    fig = go.Figure(go.Bar(
        x=drift_sorted["P-Value"],
        y=drift_sorted["Feature"],
        orientation="h",
        marker_color=[COLORS["danger"] if d else COLORS["success"] for d in drift_sorted["Drift Detected"]],
        text=[f"{p:.4f}" for p in drift_sorted["P-Value"]],
        textposition="outside",
    ))
    fig.add_vline(x=0.05, line_dash="dash", line_color=COLORS["warning"], annotation_text="α = 0.05")
    fig.update_layout(**PLOT_LAYOUT, height=max(400, len(drift_df) * 25), xaxis_title="P-Value (KS Test)", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.markdown("### Detailed Drift Report")
    display_df = drift_df[["Feature", "P-Value", "Status", "Confidence"]].copy()
    display_df["P-Value"] = display_df["P-Value"].map("{:.6f}".format)
    display_df["Confidence"] = display_df["Confidence"].map("{:.2f}%".format)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Prediction Explorer
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Prediction Explorer":
    st.markdown("# Prediction Explorer")
    st.markdown("Explore model predictions on the test dataset.")
    st.markdown("---")

    if artifact_dir is None:
        st.error("No training artifacts found.")
        st.stop()

    _, test_df = cached_load_train_test(artifact_dir)
    model = cached_load_model()
    preprocessor = cached_load_preprocessor()

    if model is None or preprocessor is None:
        st.error("Model or preprocessor not found.")
        st.stop()

    target_col = "Result"
    features = [c for c in test_df.columns if c != target_col]

    X_test = test_df[features]
    y_test = test_df[target_col]
    X_transformed = preprocessor.transform(X_test)
    y_pred = model.predict(X_transformed)

    result_df = test_df.copy()
    result_df["Prediction"] = y_pred
    # Normalize labels for comparison (handles -1/1 actuals vs 0/1 predictions)
    norm_actual, norm_pred = _normalize_labels(result_df[target_col], result_df["Prediction"])
    result_df["Correct"] = norm_actual == norm_pred
    result_df["Label_Actual"] = pd.Series(norm_actual).map(_label_name).values
    result_df["Label_Predicted"] = pd.Series(norm_pred).map(_label_name).values

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_transformed)[:, 1]
        result_df["Phishing_Probability"] = y_prob

    # Filters
    st.markdown("### Filters")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        filter_actual = st.multiselect("Actual Class:", ["Phishing", "Legitimate"], default=["Phishing", "Legitimate"])
    with fc2:
        filter_correct = st.multiselect("Prediction:", ["Correct", "Incorrect"], default=["Correct", "Incorrect"])
    with fc3:
        num_rows = st.slider("Rows to display:", 10, 500, 100)

    mask = result_df["Label_Actual"].isin(filter_actual)
    if "Correct" not in filter_correct:
        mask &= ~result_df["Correct"]
    if "Incorrect" not in filter_correct:
        mask &= result_df["Correct"]

    filtered = result_df[mask].head(num_rows)

    # Summary
    total_correct = result_df["Correct"].sum()
    total_incorrect = (~result_df["Correct"]).sum()
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Total Predictions", len(result_df))
    sc2.metric("Correct", int(total_correct))
    sc3.metric("Misclassified", int(total_incorrect))

    st.markdown("---")

    # Error analysis
    left, right = st.columns(2)
    with left:
        st.markdown("### Misclassification Breakdown")
        errors = result_df[~result_df["Correct"]]
        if len(errors) > 0:
            error_types = errors.groupby(["Label_Actual", "Label_Predicted"]).size().reset_index(name="Count")
            fig = px.bar(
                error_types, x="Count", y="Label_Actual", color="Label_Predicted",
                orientation="h",
                color_discrete_map={"Phishing": COLORS["danger"], "Legitimate": COLORS["success"]},
                labels={"Label_Actual": "Actual", "Label_Predicted": "Predicted As"},
            )
            fig.update_layout(**PLOT_LAYOUT, height=250)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No misclassifications found.")

    with right:
        if "Phishing_Probability" in result_df.columns:
            st.markdown("### Confidence Distribution")
            fig = px.histogram(
                result_df,
                x="Phishing_Probability",
                color="Correct",
                nbins=40,
                barmode="overlay",
                color_discrete_map={True: COLORS["success"], False: COLORS["danger"]},
                opacity=0.7,
            )
            fig.update_layout(**PLOT_LAYOUT, height=250, xaxis_title="Phishing Probability", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    # Data table
    st.markdown("### Prediction Details")
    display_cols = ["Label_Actual", "Label_Predicted", "Correct"]
    if "Phishing_Probability" in filtered.columns:
        display_cols.append("Phishing_Probability")
    display_cols += features[:10]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True, height=400)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Live Prediction
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Live Prediction":
    st.markdown("# Live Prediction")
    st.markdown("Upload a CSV or manually configure features to get real-time phishing predictions.")
    st.markdown("---")

    model = cached_load_model()
    preprocessor = cached_load_preprocessor()

    if model is None or preprocessor is None:
        st.error("Model or preprocessor not found. Please train the model first.")
        st.stop()

    tab_upload, tab_manual = st.tabs(["Upload CSV", "Manual input"])

    with tab_upload:
        uploaded = st.file_uploader("Upload a CSV file with feature columns", type=["csv"])
        if uploaded is not None:
            input_df = pd.read_csv(uploaded)
            st.markdown(f"**Uploaded:** {len(input_df)} rows, {len(input_df.columns)} columns")

            target_col = "Result"
            features = [c for c in input_df.columns if c != target_col and c != "predicted_column"]

            if st.button("Run predictions", type="primary"):
                X_input = input_df[features]
                X_transformed = preprocessor.transform(X_input)
                predictions = model.predict(X_transformed)
                input_df["Prediction"] = predictions
                input_df["Label"] = pd.Series(predictions).map(lambda v: "Phishing" if v in (1, 1.0) else "Legitimate").values

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_transformed)[:, 1]
                    input_df["Phishing_Probability"] = probs

                # Summary
                phish_count = (predictions == 1).sum()
                legit_count = len(predictions) - phish_count

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Total Analyzed", len(input_df))
                mc2.metric("Phishing Detected", int(phish_count))
                mc3.metric("Legitimate", int(legit_count))

                # Results chart
                fig = px.pie(
                    values=[int(phish_count), int(legit_count)],
                    names=["Phishing", "Legitimate"],
                    color_discrete_sequence=[COLORS["danger"], COLORS["success"]],
                    hole=0.45,
                )
                fig.update_layout(**PLOT_LAYOUT, height=300)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Prediction Results")
                st.dataframe(input_df, use_container_width=True, hide_index=True, height=400)

                csv = input_df.to_csv(index=False)
                st.download_button("Download results (CSV)", csv, "predictions.csv", "text/csv", type="primary")

    with tab_manual:
        st.markdown("### Configure URL Features")
        st.markdown("Set each feature value: **-1** = Legitimate, **0** = Suspicious, **1** = Phishing")

        manual_input = {}
        for cat_name, cat_features in FEATURE_CATEGORIES.items():
            st.markdown(f"**{cat_name} Features**")
            cols = st.columns(min(4, len(cat_features)))
            for i, feat in enumerate(cat_features):
                with cols[i % len(cols)]:
                    manual_input[feat] = st.selectbox(
                        feat.replace("_", " "),
                        options=[-1, 0, 1],
                        index=0,
                        key=f"manual_{feat}",
                    )

        if st.button("Analyze", type="primary"):
            input_df = pd.DataFrame([manual_input])
            X_transformed = preprocessor.transform(input_df)
            pred = model.predict(X_transformed)[0]
            label = "**PHISHING DETECTED**" if pred in (1, 1.0) else "**LEGITIMATE**"

            st.markdown("---")
            st.markdown(f"### Result: {label}")

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_transformed)[0]
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob[1] * 100,
                    title={"text": "Phishing Probability", "font": {"color": "#e2e8f0"}},
                    gauge=dict(
                        axis=dict(range=[0, 100], tickcolor="#64748b"),
                        bar=dict(color=COLORS["danger"] if pred in (1, 1.0) else COLORS["success"]),
                        bgcolor="#1e293b",
                        steps=[
                            dict(range=[0, 30], color="#064e3b"),
                            dict(range=[30, 70], color="#78350f"),
                            dict(range=[70, 100], color="#7f1d1d"),
                        ],
                        threshold=dict(
                            line=dict(color="white", width=2),
                            value=50,
                            thickness=0.8,
                        ),
                    ),
                ))
                fig.update_layout(**PLOT_LAYOUT, height=300)
                st.plotly_chart(fig, use_container_width=True)

            # Feature contribution
            importance_df = get_feature_importance(model, list(manual_input.keys()))
            if importance_df is not None:
                st.markdown("### Feature Contributions")
                top = importance_df.head(10)
                top = top.copy()
                top["value"] = top["feature"].map(manual_input)
                top["risk"] = top.apply(
                    lambda r: "High Risk" if r["value"] == 1 else ("Low Risk" if r["value"] == -1 else "Neutral"),
                    axis=1,
                )
                fig = px.bar(
                    top.iloc[::-1], x="importance", y="feature", color="risk",
                    orientation="h",
                    color_discrete_map={"High Risk": COLORS["danger"], "Low Risk": COLORS["success"], "Neutral": COLORS["warning"]},
                )
                fig.update_layout(**PLOT_LAYOUT, height=350)
                st.plotly_chart(fig, use_container_width=True)
