import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os
from datetime import datetime


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Insider Risk Detection System",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background-color: #1E222D;
        padding: 15px 20px;
        border-radius: 10px;
        border: 1px solid #2E3440;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 6px;
        background-color: #1E222D;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. MODEL ARTIFACT LOADING
# ==========================================

@st.cache_resource
def load_artifacts():
    """
    Load the trained Isolation Forest model,
    scaler, and metadata.
    """

    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    meta = joblib.load("meta.pkl")

    return model, scaler, meta


artifacts_loaded = False
model = None
scaler = None
meta = None

try:
    model, scaler, meta = load_artifacts()
    artifacts_loaded = True

except Exception as e:
    artifacts_loaded = False


# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

def calculate_risk(feature_dict):
    """
    Calculate a normalized 0-100 anomaly risk score
    for a single transaction.
    """

    if not artifacts_loaded:
        raise RuntimeError("Model artifacts are not loaded.")

    required_features = meta["features"]

    missing_features = [
        feature
        for feature in required_features
        if feature not in feature_dict
    ]

    if missing_features:
        raise ValueError(
            "The prediction is missing required model features: "
            + ", ".join(missing_features)
        )

    X = pd.DataFrame(
        [feature_dict]
    )[required_features]

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if X.isnull().any().any():
        raise ValueError(
            "One or more model features contain invalid values."
        )

    X_scaled = scaler.transform(X)

    raw_score = model.decision_function(X_scaled)[0]

    score_min = meta["score_min"]
    score_max = meta["score_max"]

    denominator = (
        score_max - score_min
    )

    if denominator == 0:
        raise ValueError(
            "Invalid score range stored in meta.pkl."
        )

    risk_score = (
        100
        * (score_max - raw_score)
        / denominator
    )

    risk_score = float(
        np.clip(
            risk_score,
            0,
            100
        )
    )

    return round(
        risk_score,
        2
    )


def get_risk_level(score):
    """
    Convert a risk score into LOW, MEDIUM, or HIGH.
    """

    if score >= 70:
        return "HIGH"

    if score >= 40:
        return "MEDIUM"

    return "LOW"


def process_batch_data(df):
    """
    Run the existing trained Isolation Forest model
    on every row of an uploaded CSV.

    The uploaded CSV must contain the same features
    that the saved model was trained on.
    """

    if not artifacts_loaded:
        raise RuntimeError(
            "Model artifacts are not available."
        )

    result_df = df.copy()

    required_features = meta["features"]

    # ------------------------------------------
    # Check required columns
    # ------------------------------------------

    missing_features = [
        feature
        for feature in required_features
        if feature not in result_df.columns
    ]

    if missing_features:
        raise ValueError(
            "The uploaded CSV is missing the following "
            "features required by the trained model:\n\n"
            + "\n".join(
                "- " + feature
                for feature in missing_features
            )
        )

    # ------------------------------------------
    # Select model features
    # ------------------------------------------

    X = result_df[
        required_features
    ].copy()

    # ------------------------------------------
    # Convert features to numeric
    # ------------------------------------------

    for column in required_features:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # ------------------------------------------
    # Check invalid or missing values
    # ------------------------------------------

    invalid_columns = X.columns[
        X.isnull().any()
    ].tolist()

    if invalid_columns:
        raise ValueError(
            "The following model features contain "
            "missing or non-numeric values:\n\n"
            + "\n".join(
                "- " + column
                for column in invalid_columns
            )
        )

    # ------------------------------------------
    # Apply saved scaler
    # ------------------------------------------

    X_scaled = scaler.transform(X)

    # ------------------------------------------
    # Run trained Isolation Forest
    # ------------------------------------------

    raw_scores = model.decision_function(
        X_scaled
    )

    # ------------------------------------------
    # Convert anomaly scores to 0-100
    # ------------------------------------------

    score_min = meta["score_min"]
    score_max = meta["score_max"]

    denominator = (
        score_max - score_min
    )

    if denominator == 0:
        raise ValueError(
            "Invalid score range stored in meta.pkl."
        )

    risk_scores = (
        100
        * (score_max - raw_scores)
        / denominator
    )

    risk_scores = np.clip(
        risk_scores,
        0,
        100
    )

    result_df["risk_score"] = np.round(
        risk_scores,
        2
    )

    # ------------------------------------------
    # Assign risk levels
    # ------------------------------------------

    result_df["risk_level"] = (
        result_df["risk_score"]
        .apply(get_risk_level)
    )

    return result_df


# ==========================================
# 5. SIDEBAR
# ==========================================

with st.sidebar:

    st.image(
        "https://img.icons8.com/color/96/shield-with-signature.png",
        width=64
    )

    st.title(
        "Insider Risk Platform"
    )

    st.caption(
        "Machine Learning Anomaly Engine v1.0"
    )

    st.divider()

    if artifacts_loaded:
        st.success(
            "ML Models Online"
        )
    else:
        st.error(
            "ML Models Offline"
        )

    st.markdown(
        "### System Metadata"
    )

    if artifacts_loaded:

        try:
            feature_count = len(
                meta["features"]
            )

            model_name = type(
                model
            ).__name__

            st.info(
                f"Model Type: {model_name}\n\n"
                f"Feature Count: {feature_count}\n\n"
                f"Target Asset: Expense Claims"
            )

        except Exception:

            st.info(
                "Model metadata could not be displayed."
            )

    else:

        st.error(
            "The model artifacts could not be loaded. "
            "Check that model.pkl, scaler.pkl, and "
            "meta.pkl are present."
        )


# ==========================================
# 6. MAIN APPLICATION
# ==========================================

st.title(
    "Enterprise Insider Risk Monitoring Dashboard"
)

st.markdown(
    "Real-time behavioral anomaly scoring and "
    "financial threat analytics powered by Isolation Forest."
)


tab1, tab2, tab3 = st.tabs(
    [
        "Risk Analytics & Overview",
        "Interactive Predictor",
        "Batch Data Upload"
    ]
)


# ==========================================
# TAB 1: EXECUTIVE ANALYTICS
# ==========================================

with tab1:

    if os.path.exists(
        "risk_results.csv"
    ):

        try:

            df_results = pd.read_csv(
                "risk_results.csv"
            )

            required_result_columns = [
                "risk_score",
                "risk_level"
            ]

            missing_result_columns = [
                column
                for column in required_result_columns
                if column not in df_results.columns
            ]

            if missing_result_columns:

                st.warning(
                    "risk_results.csv is missing required "
                    "result columns: "
                    + ", ".join(
                        missing_result_columns
                    )
                )

            else:

                # --------------------------------------
                # KPI METRICS
                # --------------------------------------

                total_txns = len(
                    df_results
                )

                high_risk_count = len(
                    df_results[
                        df_results["risk_level"]
                        == "HIGH"
                    ]
                )

                med_risk_count = len(
                    df_results[
                        df_results["risk_level"]
                        == "MEDIUM"
                    ]
                )

                avg_risk = (
                    df_results["risk_score"]
                    .mean()
                )

                col1, col2, col3, col4 = (
                    st.columns(4)
                )

                col1.metric(
                    "Total Scanned",
                    f"{total_txns:,}"
                )

                high_percentage = (
                    (high_risk_count / total_txns) * 100
                    if total_txns > 0
                    else 0
                )

                col2.metric(
                    "High-Risk Threats",
                    high_risk_count,
                    delta=f"{high_percentage:.1f}% total",
                    delta_color="inverse"
                )

                col3.metric(
                    "Medium-Risk Anomalies",
                    med_risk_count
                )

                col4.metric(
                    "Average Risk Index",
                    f"{avg_risk:.1f} / 100"
                )

                st.divider()

                # --------------------------------------
                # CHARTS
                # --------------------------------------

                col_chart1, col_chart2 = (
                    st.columns(2)
                )

                with col_chart1:

                    st.subheader(
                        "Risk Score Distribution"
                    )

                    fig_hist = px.histogram(
                        df_results,
                        x="risk_score",
                        nbins=30,
                        color="risk_level",
                        color_discrete_map={
                            "HIGH": "#EF4444",
                            "MEDIUM": "#F59E0B",
                            "LOW": "#10B981"
                        },
                        title="Density of Anomaly Scores",
                        labels={
                            "risk_score":
                                "Risk Score (0-100)",
                            "count":
                                "Transaction Count"
                        }
                    )

                    fig_hist.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )

                    st.plotly_chart(
                        fig_hist,
                        use_container_width=True
                    )

                with col_chart2:

                    st.subheader(
                        "Top Risk Employees"
                    )

                    if len(
                        df_results.columns
                    ) > 0:

                        employee_column = (
                            df_results.columns[0]
                        )

                        top_emp = (
                            df_results
                            .groupby(
                                employee_column
                            )["risk_score"]
                            .max()
                            .reset_index()
                        )

                        top_emp = (
                            top_emp
                            .sort_values(
                                by="risk_score",
                                ascending=False
                            )
                            .head(10)
                        )

                        fig_bar = px.bar(
                            top_emp,
                            x=employee_column,
                            y="risk_score",
                            color="risk_score",
                            color_continuous_scale="Reds",
                            title="Highest Individual Risk Scores",
                            labels={
                                "risk_score":
                                    "Max Risk Score"
                            }
                        )

                        fig_bar.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)"
                        )

                        st.plotly_chart(
                            fig_bar,
                            use_container_width=True
                        )

                # --------------------------------------
                # HIGH-RISK TRANSACTIONS
                # --------------------------------------

                st.subheader(
                    "Priority Incident Feed"
                )

                high_risk_df = (
                    df_results[
                        df_results["risk_level"]
                        == "HIGH"
                    ]
                    .sort_values(
                        by="risk_score",
                        ascending=False
                    )
                )

                st.dataframe(
                    high_risk_df,
                    use_container_width=True,
                    height=300
                )

        except Exception as e:

            st.error(
                "Could not load risk_results.csv."
            )

            st.code(
                str(e)
            )

    else:

        st.info(
            "risk_results.csv was not found. "
            "You can run new ML analysis from "
            "the Batch Data Upload tab."
        )


# ==========================================
# TAB 2: SINGLE TRANSACTION PREDICTOR
# ==========================================

with tab2:

    st.subheader(
        "Transaction Risk Simulator"
    )

    st.caption(
        "Enter transaction information to "
        "calculate a live anomaly score."
    )

    if not artifacts_loaded:

        st.error(
            "Prediction is unavailable because "
            "the model artifacts are missing."
        )

    else:

        with st.form(
            "risk_form"
        ):

            col_a, col_b = (
                st.columns(2)
            )

            # --------------------------------------
            # LEFT COLUMN
            # --------------------------------------

            with col_a:

                emp_id = st.text_input(
                    "Employee Identifier",
                    "EMP_042"
                )

                amount = st.number_input(
                    "Transaction Amount ($)",
                    min_value=0.0,
                    value=2850.0,
                    step=50.0
                )

                txn_date = st.date_input(
                    "Date of Transaction"
                )

                txn_time = st.time_input(
                    "Time of Transaction",
                    value=datetime.strptime(
                        "02:30",
                        "%H:%M"
                    ).time()
                )

                category_cnt = st.slider(
                    "Historical Unique Categories Used",
                    1,
                    10,
                    1
                )

            # --------------------------------------
            # RIGHT COLUMN
            # --------------------------------------

            with col_b:

                emp_avg = st.number_input(
                    "Employee Standard Average ($)",
                    min_value=0.0,
                    value=180.0
                )

                emp_std = st.number_input(
                    "Employee Standard Dev ($)",
                    min_value=0.0,
                    value=35.0
                )

                employee_txn_count = st.number_input(
                    "Employee Historical Transaction Count",
                    min_value=1,
                    value=15
                )

                daily_cnt = st.number_input(
                    "Transactions Submitted Today",
                    min_value=1,
                    value=6
                )

                mins_prev = st.number_input(
                    "Minutes Since Last Transaction",
                    min_value=0.0,
                    value=3.0
                )

            submit = st.form_submit_button(
                "Assess Risk Score",
                type="primary",
                use_container_width=True
            )

        if submit:

            try:

                # --------------------------------------
                # DERIVED FEATURES
                # --------------------------------------

                dt = pd.to_datetime(
                    f"{txn_date} {txn_time}"
                )

                hour = dt.hour

                is_weekend = (
                    1
                    if dt.dayofweek >= 5
                    else 0
                )

                unusual_hour = (
                    1
                    if (
                        hour < 6
                        or hour > 22
                    )
                    else 0
                )

                amount_dev = abs(
                    amount - emp_avg
                )

                amount_zscore = (
                    amount_dev / emp_std
                    if emp_std > 0
                    else 0.0
                )

                rapid = (
                    1
                    if mins_prev <= 15
                    else 0
                )

                # --------------------------------------
                # CREATE MODEL FEATURES
                # --------------------------------------

                feature_dict = {
                    meta["features"][0]:
                        amount,

                    "amount_deviation":
                        amount_dev,

                    "amount_zscore":
                        amount_zscore,

                    "employee_transaction_count":
                        employee_txn_count,

                    "daily_transaction_count":
                        daily_cnt,

                    "minutes_since_previous":
                        mins_prev,

                    "rapid_submission":
                        rapid,

                    "unusual_hour":
                        unusual_hour,

                    "is_weekend":
                        is_weekend,

                    "category_count":
                        category_cnt
                }

                # --------------------------------------
                # CALCULATE RISK
                # --------------------------------------

                score = calculate_risk(
                    feature_dict
                )

                level = get_risk_level(
                    score
                )

                st.divider()

                res_col1, res_col2 = (
                    st.columns([1, 2])
                )

                with res_col1:

                    if level == "HIGH":

                        st.error(
                            f"Risk Score: {score}/100\n\n"
                            f"Classification: HIGH RISK"
                        )

                    elif level == "MEDIUM":

                        st.warning(
                            f"Risk Score: {score}/100\n\n"
                            f"Classification: MEDIUM RISK"
                        )

                    else:

                        st.success(
                            f"Risk Score: {score}/100\n\n"
                            f"Classification: LOW RISK"
                        )

                # --------------------------------------
                # EXPLANATION
                # --------------------------------------

                with res_col2:

                    st.markdown(
                        "#### Threat Reason Analysis"
                    )

                    reasons = []

                    if amount_zscore >= 2.0:

                        reasons.append(
                            "Transaction amount is "
                            "significantly above the "
                            "employee historical average."
                        )

                    if unusual_hour:

                        reasons.append(
                            "Transaction was submitted "
                            "during unusual off-business hours."
                        )

                    if rapid:

                        reasons.append(
                            "Transaction occurred "
                            "within 15 minutes of the "
                            "previous transaction."
                        )

                    if daily_cnt >= 5:

                        reasons.append(
                            "High transaction velocity "
                            "was detected for the day."
                        )

                    if is_weekend:

                        reasons.append(
                            "Transaction was submitted "
                            "during the weekend."
                        )

                    if not reasons:

                        st.write(
                            "No severe rule-based anomalies "
                            "were detected from the supplied "
                            "transaction information."
                        )

                    else:

                        for reason in reasons:

                            st.write(
                                "- " + reason
                            )

            except Exception as e:

                st.error(
                    "The transaction could not be "
                    "processed by the model."
                )

                st.code(
                    str(e)
                )


# ==========================================
# TAB 3: BATCH CSV ML ANALYSIS
# ==========================================

with tab3:

    st.subheader(
        "Batch ML Risk Detection"
    )

    st.caption(
        "Upload a CSV file and run the trained "
        "Isolation Forest model on every transaction."
    )

    if not artifacts_loaded:

        st.error(
            "Batch prediction is unavailable because "
            "the model artifacts are missing."
        )

    else:

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"]
        )

        if uploaded_file is not None:

            try:

                # --------------------------------------
                # LOAD CSV
                # --------------------------------------

                user_df = pd.read_csv(
                    uploaded_file
                )

                st.success(
                    f"File loaded successfully: "
                    f"{len(user_df):,} rows."
                )

                st.write(
                    "### Uploaded Data Preview"
                )

                st.dataframe(
                    user_df.head(10),
                    use_container_width=True
                )

                # --------------------------------------
                # SHOW EXPECTED FEATURES
                # --------------------------------------

                with st.expander(
                    "View features expected by the trained model"
                ):

                    st.write(
                        meta["features"]
                    )

                # --------------------------------------
                # RUN MODEL
                # --------------------------------------

                if st.button(
                    "Run ML Risk Detection",
                    type="primary",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Running the trained Isolation Forest model..."
                    ):

                        try:

                            results = process_batch_data(
                                user_df
                            )

                            st.success(
                                "ML analysis completed successfully."
                            )

                            # ----------------------------------
                            # SUMMARY
                            # ----------------------------------

                            st.subheader(
                                "Risk Summary"
                            )

                            high_count = len(
                                results[
                                    results["risk_level"]
                                    == "HIGH"
                                ]
                            )

                            medium_count = len(
                                results[
                                    results["risk_level"]
                                    == "MEDIUM"
                                ]
                            )

                            low_count = len(
                                results[
                                    results["risk_level"]
                                    == "LOW"
                                ]
                            )

                            average_score = (
                                results["risk_score"]
                                .mean()
                            )

                            col1, col2, col3, col4 = (
                                st.columns(4)
                            )

                            col1.metric(
                                "High Risk",
                                high_count
                            )

                            col2.metric(
                                "Medium Risk",
                                medium_count
                            )

                            col3.metric(
                                "Low Risk",
                                low_count
                            )

                            col4.metric(
                                "Average Risk",
                                f"{average_score:.1f}"
                            )

                            # ----------------------------------
                            # DISTRIBUTION
                            # ----------------------------------

                            st.subheader(
                                "Batch Risk Distribution"
                            )

                            fig = px.histogram(
                                results,
                                x="risk_score",
                                nbins=30,
                                color="risk_level",
                                color_discrete_map={
                                    "HIGH": "#EF4444",
                                    "MEDIUM": "#F59E0B",
                                    "LOW": "#10B981"
                                },
                                labels={
                                    "risk_score":
                                        "Risk Score (0-100)",
                                    "count":
                                        "Transactions"
                                }
                            )

                            fig.update_layout(
                                template="plotly_dark",
                                paper_bgcolor=
                                    "rgba(0,0,0,0)",
                                plot_bgcolor=
                                    "rgba(0,0,0,0)"
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

                            # ----------------------------------
                            # RESULTS
                            # ----------------------------------

                            st.subheader(
                                "Highest Risk Transactions"
                            )

                            sorted_results = (
                                results
                                .sort_values(
                                    by="risk_score",
                                    ascending=False
                                )
                            )

                            st.dataframe(
                                sorted_results,
                                use_container_width=True,
                                height=450
                            )

                            # ----------------------------------
                            # DOWNLOAD
                            # ----------------------------------

                            output_csv = (
                                sorted_results
                                .to_csv(
                                    index=False
                                )
                                .encode("utf-8")
                            )

                            st.download_button(
                                label=
                                    "Download ML Risk Results",
                                data=output_csv,
                                file_name=
                                    "insider_risk_results.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                        except ValueError as e:

                            st.error(
                                "The uploaded dataset "
                                "cannot be processed."
                            )

                            st.warning(
                                str(e)
                            )

                        except Exception as e:

                            st.error(
                                "An unexpected error occurred "
                                "while running the ML model."
                            )

                            st.code(
                                str(e)
                            )

            except Exception as e:

                st.error(
                    "The CSV file could not be read."
                )

                st.code(
                    str(e)
                )
