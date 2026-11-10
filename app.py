
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NIDS Pro | Network Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "nids_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
DATASET_PATH = BASE_DIR / "dataset" / "network_data.csv"


# --------------------------------------------------
# CUSTOM DARK BLUE DESIGN
# --------------------------------------------------

st.markdown("""
<style>
    @import url(
      'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #08111f;
        color: #e8f0fc;
    }

    [data-testid="stSidebar"] {
        background: #0c192b;
        border-right: 1px solid #1e3552;
    }

    [data-testid="stMetric"] {
        background: #101f34;
        border: 1px solid #203b5c;
        padding: 18px;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #9eb5d1;
    }

    [data-testid="stMetricValue"] {
        color: #60a5fa;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #eaf3ff;
        margin-bottom: 0;
    }

    .sub-title {
        color: #8da9c9;
        font-size: 14px;
        margin-top: 5px;
    }

    .section-title {
        color: #dceaff;
        font-size: 20px;
        font-weight: 700;
        margin-top: 20px;
    }

    .status-card {
        background: linear-gradient(
            135deg, #102744, #102039
        );
        border: 1px solid #28517e;
        border-radius: 14px;
        padding: 20px;
        margin: 10px 0 20px 0;
    }

    .status-text {
        color: #60a5fa;
        font-size: 14px;
        font-weight: 700;
    }

    div.stButton > button {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background: #1d4ed8;
        color: white;
        border: none;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    hr {
        border-color: #20354f;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD TRAINED MODEL AND METRICS
# --------------------------------------------------

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return None


bundle = load_model()
metrics = load_metrics()


# --------------------------------------------------
# COMMON HEADER
# --------------------------------------------------

st.markdown(
    '<p class="main-title">🛡️ NIDS Pro</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="sub-title">'
    'AI-Powered Network Intrusion Detection System'
    ' | Security Intelligence Platform'
    '</p>',
    unsafe_allow_html=True,
)

st.divider()


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🛡️ NIDS Pro")
    st.caption("NETWORK SECURITY INTELLIGENCE")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Threat Prediction",
            "Model Performance",
            "Dataset Analysis",
            "About Project",
        ],
    )

    st.divider()

    if bundle:
        st.success("AI Model Loaded")
        st.caption(f"Active model: {bundle['model_name']}")
    else:
        st.warning("Model not trained yet")
        st.caption("Run train_model.py first.")

    st.caption("Phase 1 · AI Detection Engine")


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def styled_chart(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="#101f34",
        plot_bgcolor="#101f34",
        font_color="#dceaff",
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c7d9ee"),
        ),
    )

    fig.update_xaxes(
        gridcolor="#20354f",
        zerolinecolor="#20354f",
    )

    fig.update_yaxes(
        gridcolor="#20354f",
        zerolinecolor="#20354f",
    )

    return fig


def prepare_features(data, feature_columns):
    data = data.copy()

    # Clean CSV headers.
    data.columns = [str(c).strip() for c in data.columns]

    # Create missing model features.
    for column in feature_columns:
        if column not in data.columns:
            data[column] = np.nan

    # Keep only the features used during training.
    data = data[feature_columns].copy()

    for column in data.columns:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.replace([np.inf, -np.inf], np.nan)

    return data


def predict_traffic(data):
    if bundle is None:
        st.error("No trained model found. Train the model first.")
        return None

    feature_columns = bundle["feature_columns"]
    class_names = bundle["class_names"]
    model = bundle["model"]

    features = prepare_features(data, feature_columns)

    if features.empty:
        st.error("The uploaded file has no records.")
        return None

    predictions = model.predict(features).astype(int)

    results = data.copy()

    results["Predicted_Attack"] = [
        class_names[index] for index in predictions
    ]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)

        results["Confidence_Percent"] = (
            probabilities.max(axis=1) * 100
        ).round(2)

    results["Threat_Status"] = [
        "NORMAL" if name.upper() == "BENIGN" else "THREAT"
        for name in results["Predicted_Attack"]
    ]

    return results


# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

if page == "Dashboard":

    st.markdown("## Security Overview")

    if metrics is None or bundle is None:

        st.warning(
            "The AI engine is not ready. "
            "Run the training script first."
        )

        st.code("python train_model.py", language="powershell")

    else:

        if metrics.get("warning"):
            st.warning(metrics["warning"])

        st.markdown(
            '<div class="status-card">'
            '<span class="status-text">● AI ENGINE OPERATIONAL</span>'
            '<br><br>'
            '<b>Machine-learning model is loaded and ready '
            'to classify uploaded network traffic.</b>'
            '</div>',
            unsafe_allow_html=True,
        )

        best_name = metrics["best_model"]
        best = metrics["models"][best_name]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Dataset Records",
            f"{metrics['dataset_rows']:,}"
        )

        col2.metric(
            "Network Features",
            metrics["feature_count"]
        )

        col3.metric(
            "Attack Classes",
            len(metrics["class_names"])
        )

        col4.metric(
            "Best Model",
            best_name
        )

        st.markdown("### Model Performance")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Accuracy",
            f"{best['accuracy'] * 100:.2f}%"
        )

        c2.metric(
            "Precision",
            f"{best['precision'] * 100:.2f}%"
        )

        c3.metric(
            "Weighted F1 Score",
            f"{best['f1_score'] * 100:.2f}%"
        )

        st.markdown("### Model Comparison")

        comparison_data = []

        for name, result in metrics["models"].items():
            comparison_data.append({
                "Model": name,
                "Accuracy": result["accuracy"] * 100,
                "Precision": result["precision"] * 100,
                "Recall": result["recall"] * 100,
                "F1 Score": result["f1_score"] * 100,
            })

        comparison_df = pd.DataFrame(comparison_data)

        fig = px.bar(
            comparison_df,
            x="Model",
            y=["Accuracy", "Precision", "Recall", "F1 Score"],
            barmode="group",
            title="Classifier Evaluation Comparison",
        )

        st.plotly_chart(
            styled_chart(fig),
            use_container_width=True,
        )

        st.caption(
            "Scores are measured on the held-out test split. "
            "They do not represent live network performance."
        )


# --------------------------------------------------
# THREAT PREDICTION PAGE
# --------------------------------------------------

elif page == "Threat Prediction":

    st.markdown("## 🔍 Network Threat Prediction")

    st.write(
        "Upload a CSV containing network traffic features. "
        "The trained AI model will classify each record."
    )

    if bundle is None:
        st.warning("Train the AI model before using prediction.")
        st.code("python train_model.py")

    else:
        uploaded_file = st.file_uploader(
            "Upload Network Traffic CSV",
            type=["csv"],
        )

        if uploaded_file is not None:

            try:
                traffic_df = pd.read_csv(uploaded_file)

                st.markdown("### Uploaded Traffic Preview")
                st.dataframe(
                    traffic_df.head(10),
                    use_container_width=True,
                )

                if st.button(
                    "🚀 Analyze Network Traffic",
                    use_container_width=True,
                ):

                    with st.spinner(
                        "AI engine analyzing network traffic..."
                    ):

                        results = predict_traffic(traffic_df)

                    if results is not None:

                        st.success("Traffic analysis completed!")

                        total = len(results)

                        threats = int(
                            (results["Threat_Status"] == "THREAT").sum()
                        )

                        benign = total - threats

                        c1, c2, c3 = st.columns(3)

                        c1.metric("Total Records", total)
                        c2.metric("Threat Records", threats)
                        c3.metric("Benign Records", benign)

                        st.markdown("### Prediction Results")

                        st.dataframe(
                            results,
                            use_container_width=True,
                        )

                        chart_data = pd.DataFrame({
                            "Status": ["Threat", "Benign"],
                            "Count": [threats, benign],
                        })

                        fig = px.pie(
                            chart_data,
                            names="Status",
                            values="Count",
                            title="Traffic Classification",
                            hole=0.55,
                        )

                        st.plotly_chart(
                            styled_chart(fig),
                            use_container_width=True,
                        )

                        csv_data = results.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            "⬇️ Download Prediction Report",
                            data=csv_data,
                            file_name="nids_prediction_report.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

            except Exception as error:
                st.error(f"Could not process CSV: {error}")


# --------------------------------------------------
# MODEL PERFORMANCE PAGE
# --------------------------------------------------

elif page == "Model Performance":

    st.markdown("## 📊 Model Evaluation")

    if metrics is None:
        st.warning("No evaluation results found.")
        st.code("python train_model.py")

    else:
        st.write(
            f"Selected model: **{metrics['best_model']}**"
        )

        selected_model = st.selectbox(
            "Choose a trained model",
            list(metrics["models"].keys()),
        )

        result = metrics["models"][selected_model]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Accuracy", f"{result['accuracy']:.3f}")
        c2.metric("Precision", f"{result['precision']:.3f}")
        c3.metric("Recall", f"{result['recall']:.3f}")
        c4.metric("F1 Score", f"{result['f1_score']:.3f}")

        st.markdown("### Confusion Matrix")

        matrix = np.array(result["confusion_matrix"])
        labels = metrics["class_names"]

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=labels,
                y=labels,
                colorscale="Blues",
                text=matrix,
                texttemplate="%{text}",
                hoverongaps=False,
            )
        )

        fig.update_layout(
            title="Actual vs Predicted Classes",
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class",
        )

        st.plotly_chart(
            styled_chart(fig),
            use_container_width=True,
        )

        st.markdown("### Classification Report")

        report = pd.DataFrame(
            result["classification_report"]
        ).T

        st.dataframe(
            report.round(3),
            use_container_width=True,
        )

        if metrics.get("feature_importance"):
            st.markdown("### Important Network Features")

            importance_df = pd.DataFrame(
                list(metrics["feature_importance"].items()),
                columns=["Feature", "Importance"],
            )

            fig = px.bar(
                importance_df.head(15).sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top Features Used by Selected Model",
            )

            st.plotly_chart(
                styled_chart(fig),
                use_container_width=True,
            )


# --------------------------------------------------
# DATASET ANALYSIS PAGE
# --------------------------------------------------

elif page == "Dataset Analysis":

    st.markdown("## 📁 Dataset Analysis")

    uploaded_dataset = st.file_uploader(
        "Upload a dataset for inspection",
        type=["csv"],
        key="dataset_analysis",
    )

    if uploaded_dataset is not None:

        try:
            df = pd.read_csv(uploaded_dataset)

            c1, c2, c3 = st.columns(3)

            c1.metric("Rows", f"{len(df):,}")
            c2.metric("Columns", len(df.columns))
            c3.metric(
                "Missing Values",
                f"{int(df.isna().sum().sum()):,}"
            )

            st.markdown("### Dataset Preview")
            st.dataframe(
                df.head(20),
                use_container_width=True,
            )

            st.markdown("### Column Information")

            info = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Missing Values": df.isna().sum().values,
                "Unique Values": df.nunique().values,
            })

            st.dataframe(
                info,
                use_container_width=True,
            )

            st.markdown("### Numeric Feature Distribution")

            numeric_columns = df.select_dtypes(
                include=np.number
            ).columns.tolist()

            if numeric_columns:

                selected_feature = st.selectbox(
                    "Select a feature",
                    numeric_columns,
                )

                fig = px.histogram(
                    df,
                    x=selected_feature,
                    nbins=40,
                    title=f"Distribution of {selected_feature}",
                )

                st.plotly_chart(
                    styled_chart(fig),
                    use_container_width=True,
                )

            else:
                st.info("No numeric columns found.")

        except Exception as error:
            st.error(f"Dataset error: {error}")

    elif DATASET_PATH.exists():

        st.info(
            "Upload a CSV above to explore it. "
            "The training dataset is available in the project folder."
        )

    else:
        st.info("Upload a CSV file to start dataset analysis.")


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

elif page == "About Project":

    st.markdown("## About NIDS Pro")

    st.write("""
    **NIDS Pro** is an AI-based Network Intrusion Detection System
    developed as an advanced final-year engineering project.

    Its objective is to classify network traffic and identify
    potentially malicious activity using supervised machine learning.
    """)

    st.markdown("### Technology Stack")

    technologies = [
        "Python",
        "Pandas and NumPy",
        "Scikit-learn",
        "XGBoost",
        "Streamlit",
        "Plotly",
        "Joblib",
    ]

    for technology in technologies:
        st.markdown(f"- {technology}")

    st.markdown("### Phase 1 Features")

    st.markdown("""
    - Network traffic dataset preprocessing
    - Random Forest model training
    - XGBoost model training
    - Model comparison
    - Classification metrics
    - Confusion matrix
    - Feature importance
    - CSV-based threat prediction
    - Downloadable prediction results
    """)

    st.info(
        "Phase 1 analyzes uploaded network-flow records. "
        "It does not yet capture live network packets."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "NIDS Pro | AI Network Intrusion Detection System | "
    "Phase 1 Development"
)
