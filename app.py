from datetime import datetime
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION & EXECUTIVE THEME
# ==========================================
st.set_page_config(
    page_title="Insider Risk Detection System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Professional Dark Enterprise Aesthetic
st.markdown(
    """
<style>
    /* Metric Card Styling */
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
    /* Tab Styling */
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
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2. ARTIFACT LOADING ENGINE
# ==========================================
@st.cache_resource
def load_artifacts():
    """Loads trained Isolation Forest models and metadata."""
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    meta = joblib.load("meta.pkl")
    return model, scaler, meta


try:
    model, scaler, meta = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    st.sidebar.error(f"Model Artifact Error: {e}")


# ==========================================
# 3. HELPER PREDICTION FUNCTIONS
# ==========================================
def calculate_risk(feature_dict):
    """Calculates normalized 0-100 risk score using loaded models."""
    X = pd.DataFrame([feature_dict])[meta["features"]]
    X_scaled = scaler.transform(X)

    raw_score = model.decision_function(X_scaled)[0]
    score_min, score_max = meta["score_min"], meta["score_max"]

    risk_score = round(
        float(
            np.clip(
                100 * (score_max - raw_score) / (score_max - score_min + 1e-9),
                0,
                100,
            )
        ),
        2,
    )

    return risk_score


def get_risk_level(score):
    if score >= 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    return "LOW"


# ==========================================
# 4. SIDEBAR NAVIGATION & CONTROLS
# ==========================================
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/shield-with-signature.png", width=64
    )

    st.title("Insider Risk Platform")
    st.caption("Machine Learning Anomaly Engine v1.0")

    st.divider()

    if artifacts_loaded:
        st.success("ML Models Online")
    else:
        st.error("Models Offline")

    st.markdown("### System Metadata")

    st.info(
        f"**Model Type:** Isolation Forest\n\n"
        f"**Feature Count:** "
        f"{len(meta['features']) if artifacts_loaded else 0}\n\n"
        f"**Target Asset:** Expense Claims"
    )


# ==========================================
# 5. MAIN DASHBOARD CONTENT
# ==========================================
st.title("Enterprise Insider Risk Monitoring Dashboard")

st.markdown(
    "Real-time behavioral anomaly scoring and "
    "financial threat analytics powered by Isolation Forest."
)

tab1, tab2, tab3 = st.tabs(
    ["Risk Analytics & Overview", "Interactive Predictor", "Batch Data Upload"]
)


# ------------------------------------------
# TAB 1: EXECUTIVE ANALYTICS
# ------------------------------------------
with tab1:
    if os.path.exists("risk_results.csv"):
        df_results = pd.read_csv("risk_results.csv")

        col1, col2, col3, col4 = st.columns(4)

        total_txns = len(df_results)
        high_risk_count = len(df_results[df_results["risk_level"] == "HIGH"])
        med_risk_count = len(df_results[df_results["risk_level"] == "MEDIUM"])
        avg_risk = df_results["risk_score"].mean()

        col1.metric("Total Scanned", f"{total_txns:,}")
        col2.metric(
            "High-Risk Threats",
            high_risk_count,
            delta=f"{(high_risk_count / total_txns) * 100:.1f}% total",
            delta_color="inverse",
        )
        col3.metric("Medium-Risk Anomalies", med_risk_count)
        col4.metric("Average Risk Index", f"{avg_risk:.1f} / 100")

        st.divider()

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.subheader("Risk Score Distribution")

            fig_hist = px.histogram(
                df_results,
                x="risk_score",
                nbins=30,
                color="risk_level",
                color_discrete_map={
                    "HIGH": "#EF4444",
                    "MEDIUM": "#F59E0B",
                    "LOW": "#10B981",
                },
                title="Density of Anomaly Scores",
                labels={
                    "risk_score": "Risk Score (0-100)",
                    "count": "Transaction Count",
                },
            )

            fig_hist.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig_hist, use_container_width=True)

        with col_chart2:
            st.subheader("Top Risk Employees")

            top_emp = (
                df_results.groupby(df_results.columns[0])["risk_score"]
                .max()
                .reset_index()
            )

            top_emp = top_emp.sort_values(
                by="risk_score", ascending=False
            ).head(10)

            fig_bar = px.bar(
                top_emp,
                x=top_emp.columns[0],
                y="risk_score",
                color="risk_score",
                color_continuous_scale="Reds",
                title="Highest Individual Risk Scores",
                labels={"risk_score": "Max Risk Score"},
            )

            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Priority Incident Feed (High-Risk Submissions)")

        high_risk_df = df_results[
            df_results["risk_level"] == "HIGH"
        ].sort_values(by="risk_score", ascending=False)

        st.dataframe(
            high_risk_df,
            use_container_width=True,
            height=300,
        )

    else:
        st.warning(
            "`risk_results.csv` not found in current directory. "
            "Run model training or upload batch CSV in Tab 3."
        )


# ------------------------------------------
# TAB 2: LIVE SINGLE PREDICTOR
# ------------------------------------------
with tab2:
    st.subheader("Transaction Risk Simulator")
    st.caption(
        "Input employee transaction details to calculate live anomaly scores."
    )

    if not artifacts_loaded:
        st.error("Cannot perform prediction: Model artifacts are missing.")
    else:
        with st.form("risk_form"):
            col_a, col_b = st.columns(2)

            with col_a:
                emp_id = st.text_input("Employee Identifier", "EMP_042")
                amount = st.number_input(
                    "Transaction Amount ($)",
                    min_value=0.0,
                    value=2850.0,
                    step=50.0,
                )
                txn_date = st.date_input("Date of Transaction")
                txn_time = st.time_input(
                    "Time of Transaction",
                    value=datetime.strptime("02:30", "%H:%M").time(),
                )
                category_cnt = st.slider(
                    "Historical Unique Categories Used", 1, 10, 1
                )

            with col_b:
                emp_avg = st.number_input(
                    "Employee Standard Average ($)", min_value=0.0, value=180.0
                )
                emp_std = st.number_input(
                    "Employee Standard Dev ($)", min_value=0.0, value=35.0
                )
                emp_txn_cnt = st.number_input(
                    "Employee Historical Transaction Count",
                    min_value=0,
                    value=15,
                )
                daily_cnt = st.number_input(
                    "Transactions Submitted Today", min_value=1, value=6
                )
                mins_prev = st.number_input(
                    "Minutes Since Last Transaction",
                    min_value=0.0,
                    value=3.0,
                )

            submit = st.form_submit_button(
                "Assess Risk Score", type="primary", use_container_width=True
            )

        if submit:
            dt = pd.to_datetime(f"{txn_date} {txn_time}")
            hour = dt.hour
            is_weekend = 1 if dt.dayofweek >= 5 else 0
            unusual_hour = 1 if (hour < 6 or hour > 22) else 0
            amount_dev = abs(amount - emp_avg)
            amount_zscore = (amount_dev / emp_std) if emp_std > 0 else 0.0
            rapid = 1 if mins_prev <= 15 else 0

            feature_dict = {
                "amount": amount,
                "amount_deviation": amount_dev,
                "amount_zscore": amount_zscore,
                "employee_transaction_count": emp_txn_cnt,
                "daily_transaction_count": daily_cnt,
                "minutes_since_previous": mins_prev,
                "rapid_submission": rapid,
                "unusual_hour": unusual_hour,
                "is_weekend": is_weekend,
                "category_count": category_cnt,
            }

            score = calculate_risk(feature_dict)
            level = get_risk_level(score)

            st.divider()

            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                if level == "HIGH":
                    st.error(
                        f"Risk Score: {score}/100\nClassification: HIGH RISK"
                    )
                elif level == "MEDIUM":
                    st.warning(
                        f"Risk Score: {score}/100\nClassification: MEDIUM RISK"
                    )
                else:
                    st.success(
                        f"Risk Score: {score}/100\nClassification: LOW RISK"
                    )

            with res_col2:
                st.markdown("#### Threat Reason Analysis")

                reasons = []
                if amount_zscore >= 2.0:
                    reasons.append(
                        "Transaction amount vastly exceeds employee historical average."
                    )
                if unusual_hour:
                    reasons.append(
                        "Submission logged during unusual off-business hours."
                    )
                if rapid:
                    reasons.append(
                        "Rapid succession transaction (< 15 min gap)."
                    )
                if daily_cnt >= 5:
                    reasons.append("High daily transaction velocity detected.")
                if is_weekend:
                    reasons.append("Transaction logged during the weekend.")

                if not reasons:
                    st.write(
                        "No severe anomalies detected. Transaction fits standard profile."
                    )
                else:
                    for r in reasons:
                        st.write(f"- {r}")


# ------------------------------------------
# TAB 3: BATCH DATA UPLOAD & INFERENCE ENGINE
# ------------------------------------------
with tab3:
    st.subheader("Batch Inference Engine & Anomaly Pipeline")
    st.caption(
        "Upload a dataset matching the training schema. The engine applies exact time-series "
        "and behavioral transformations before evaluating Isolation Forest anomaly scores."
    )

    if "uploaded_file_id" not in st.session_state:
        st.session_state["uploaded_file_id"] = None

    uploaded_file = st.file_uploader(
        "Upload Evaluation Dataset (CSV)", type=["csv"]
    )

    if uploaded_file is not None:
        # Check against unique Streamlit file_id to detect new uploads
        if st.session_state["uploaded_file_id"] != uploaded_file.file_id:
            st.session_state["uploaded_file_id"] = uploaded_file.file_id
            if "batch_results" in st.session_state:
                del st.session_state["batch_results"]

        try:
            raw_df = pd.read_csv(uploaded_file)
            st.write("### Raw Uploaded Preview")
            st.dataframe(raw_df.head(3), use_container_width=True)

            run_pipeline = st.button(
                "Run Model Inference Pipeline", type="primary"
            )

            if run_pipeline or "batch_results" in st.session_state:
                if not artifacts_loaded:
                    st.error(
                        "Model artifacts (model.pkl, scaler.pkl, meta.pkl) are not loaded."
                    )
                else:
                    if run_pipeline:
                        processed_df = raw_df.copy()

                        # Normalize column lookup names
                        col_map = {
                            col.lower().strip(): col for col in raw_df.columns
                        }

                        # Find and explicitly map column aliases to canonical feature names
                        amt_col = next(
                            (
                                col_map[c]
                                for c in [
                                    "amount",
                                    "txn_amount",
                                    "transaction_amount",
                                ]
                                if c in col_map
                            ),
                            None,
                        )
                        if amt_col and "amount" not in processed_df.columns:
                            processed_df["amount"] = processed_df[amt_col]

                        emp_col = next(
                            (
                                col_map[c]
                                for c in ["employee_id", "emp_id", "user_id"]
                                if c in col_map
                            ),
                            None,
                        )
                        if emp_col and "employee_id" not in processed_df.columns:
                            processed_df["employee_id"] = processed_df[emp_col]

                        # Temporal Features
                        time_col = next(
                            (
                                col_map[c]
                                for c in [
                                    "timestamp",
                                    "date",
                                    "txn_date",
                                    "datetime",
                                ]
                                if c in col_map
                            ),
                            None,
                        )
                        if time_col:
                            dt_series = pd.to_datetime(
                                processed_df[time_col], errors="coerce"
                            )
                            if "unusual_hour" not in processed_df.columns:
                                processed_df["unusual_hour"] = (
                                    dt_series.dt.hour.apply(
                                        lambda h: (
                                            1
                                            if (h < 6 or h > 22)
                                            else 0 if pd.notnull(h) else 0
                                        )
                                    )
                                )
                            if "is_weekend" not in processed_df.columns:
                                processed_df["is_weekend"] = (
                                    dt_series.dt.dayofweek.apply(
                                        lambda d: (
                                            1
                                            if d >= 5
                                            else 0 if pd.notnull(d) else 0
                                        )
                                    )
                                )

                        # Behavioral Features
                        target_amt = "amount" if "amount" in processed_df.columns else amt_col
                        target_emp = "employee_id" if "employee_id" in processed_df.columns else emp_col

                        if target_amt:
                            processed_df[target_amt] = pd.to_numeric(
                                processed_df[target_amt], errors="coerce"
                            )

                            if (
                                "amount_deviation"
                                not in processed_df.columns
                            ):
                                if target_emp:
                                    emp_means = processed_df.groupby(target_emp)[
                                        target_amt
                                    ].transform("mean")
                                    processed_df["amount_deviation"] = (
                                        processed_df[target_amt] - emp_means
                                    ).abs()
                                else:
                                    processed_df["amount_deviation"] = (
                                        processed_df[target_amt]
                                        - processed_df[target_amt].mean()
                                    ).abs()

                            if "amount_zscore" not in processed_df.columns:
                                if target_emp:
                                    emp_stds = (
                                        processed_df.groupby(target_emp)[target_amt]
                                        .transform("std")
                                        .replace(0, 1.0)
                                        .fillna(1.0)
                                    )
                                    emp_means = processed_df.groupby(target_emp)[
                                        target_amt
                                    ].transform("mean")
                                    processed_df["amount_zscore"] = (
                                        (processed_df[target_amt] - emp_means).abs()
                                        / emp_stds
                                    ).fillna(0.0)
                                else:
                                    pop_std = processed_df[target_amt].std() or 1.0
                                    processed_df["amount_zscore"] = (
                                        (
                                            processed_df[target_amt]
                                            - processed_df[target_amt].mean()
                                        ).abs()
                                        / pop_std
                                    ).fillna(0.0)

                        if (
                            "rapid_submission" not in processed_df.columns
                            and "minutes_since_previous" in processed_df.columns
                        ):
                            processed_df["rapid_submission"] = processed_df[
                                "minutes_since_previous"
                            ].apply(lambda m: 1 if m <= 15 else 0)

                        # Schema Validation
                        required_features = meta["features"]
                        missing_features = [
                            f
                            for f in required_features
                            if f not in processed_df.columns
                        ]

                        if missing_features:
                            st.error(
                                f"**Pipeline Validation Failed:** Dataset is missing required feature columns "
                                f"that cannot be derived: `{missing_features}`."
                            )
                        else:
                            X = processed_df[required_features].copy()
                            X = X.apply(pd.to_numeric, errors="coerce")

                            # Strict Missing Value Check (No silent zero-filling)
                            if X.isnull().any().any():
                                invalid_features = X.columns[
                                    X.isnull().any()
                                ].tolist()
                                st.error(
                                    f"Invalid or missing numeric values found in: `{invalid_features}`"
                                )
                            else:
                                X_scaled = scaler.transform(X)
                                raw_scores = model.decision_function(X_scaled)

                                score_min = meta["score_min"]
                                score_max = meta["score_max"]

                                risk_scores = np.clip(
                                    100
                                    * (score_max - raw_scores)
                                    / (score_max - score_min + 1e-9),
                                    0,
                                    100,
                                )

                                output_df = raw_df.copy()
                                output_df["risk_score"] = np.round(
                                    risk_scores, 2
                                )
                                output_df["risk_level"] = output_df[
                                    "risk_score"
                                ].apply(get_risk_level)

                                st.session_state["batch_results"] = output_df

                    if "batch_results" in st.session_state:
                        results_df = st.session_state["batch_results"]

                        high_count = len(
                            results_df[results_df["risk_level"] == "HIGH"]
                        )
                        med_count = len(
                            results_df[results_df["risk_level"] == "MEDIUM"]
                        )
                        low_count = len(
                            results_df[results_df["risk_level"] == "LOW"]
                        )

                        st.success(
                            f"Successfully processed {len(results_df):,} records through the ML pipeline."
                        )

                        m1, m2, m3 = st.columns(3)
                        m1.metric("High-Risk Anomaly Count", high_count)
                        m2.metric("Medium-Risk Warning Count", med_count)
                        m3.metric("Low-Risk Normal Count", low_count)

                        st.divider()

                        st.subheader("Scored Evaluation Dataset")
                        results_sorted = results_df.sort_values(
                            by="risk_score", ascending=False
                        )
                        st.dataframe(results_sorted, use_container_width=True)

                        csv_data = results_sorted.to_csv(index=False).encode(
                            "utf-8"
                        )
                        st.download_button(
                            label="Download Scored Results CSV",
                            data=csv_data,
                            file_name="insider_risk_predictions.csv",
                            mime="text/csv",
                        )

        except Exception as e:
            st.error(f"Dataset Processing Error: {str(e)}")
