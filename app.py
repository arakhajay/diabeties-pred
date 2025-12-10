import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Definition of the missing class to allow joblib to load the object
class EliteXGBoostPredictor:
    def __init__(self):
        self.model = None
        self.production_threshold = 0.5
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- MODEL LOADING ----------
@st.cache_resource
def load_model():
    return joblib.load("elite_xgboost_production.joblib")

xgb_model = load_model()

# ---------- SESSION STATE INIT ----------
if "history" not in st.session_state:
    st.session_state["history"] = []

if "current_prediction" not in st.session_state:
    st.session_state["current_prediction"] = None

# ---------- STYLES ----------
st.markdown(
    """
<style>
.main-header {
    font-size: 2.5rem;
    color: #4facfe;
    text-align: center;
    margin-bottom: 1.5rem;
    font-weight: bold;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.2rem;
    border-radius: 1rem;
    color: white;
    text-align: center;
}
.prediction-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 1.5rem;
    border-radius: 1.5rem;
    text-align: center;
    margin-top: 1rem;
}
.history-card {
    background: #1f2933;
    padding: 0.6rem 0.8rem;
    border-radius: 0.6rem;
    margin-bottom: 0.5rem;
    color: #e5e7eb;
    font-size: 0.85rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 🩺 Patient Info")
    patient_name = st.text_input("Patient Name", placeholder="Enter patient name")

    st.markdown("---")
    if st.button("🗑️ Clear History"):
        st.session_state["history"] = []
        st.session_state["current_prediction"] = None

    st.markdown("### 📋 Prediction History")
    if st.session_state["history"]:
        # show latest first
        for rec in reversed(st.session_state["history"][-10:]):
            st.markdown(
                f"""
                <div class="history-card">
                    <b>{rec['name']}</b><br/>
                    {rec['result']} | {rec['probability']}<br/>
                    {rec['timestamp']}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No predictions yet. Make your first prediction!")

# ---------- MAIN HEADER ----------
st.markdown(
    '<h1 class="main-header">🩺 Diabetes Predictor</h1>',
    unsafe_allow_html=True,
)

top_c1, top_c2 = st.columns([1, 2])
with top_c2:
    st.markdown(
        """
        <div class="metric-card">
            <h4>🔬 AI-Powered Diabetes Risk Estimation</h4>
            <p>Enter patient details and get an instant prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- INPUT FORM ----------
st.markdown("## 📝 Enter Patient Details")
c1, c2 = st.columns(2)

with c1:
    age = st.number_input("👴 Age", 0, 120, 30)
    hypertension = st.selectbox(
        "💉 Hypertension (0/1)",
        [0, 1],
        format_func=lambda x: "0 - No" if x == 0 else "1 - Yes",
    )
    heart_disease = st.selectbox(
        "❤️ Heart Disease (0/1)",
        [0, 1],
        format_func=lambda x: "0 - No" if x == 0 else "1 - Yes",
    )
    bmi = st.number_input("⚖️ BMI", 0.0, 100.0, 25.0, step=0.1)

with c2:
    HbA1c_level = st.number_input("🩸 HbA1c Level", 0.0, 20.0, 5.5, step=0.1)
    blood_glucose_level = st.number_input(
        "🌡️ Blood Glucose (mg/dL)", 0.0, 500.0, 100.0
    )
    smoking_history_former = st.selectbox(
        "🚬 Former Smoker (0/1)",
        [0, 1],
        format_func=lambda x: "0 - No" if x == 0 else "1 - Yes",
    )

input_data = pd.DataFrame(
    {
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "bmi": [bmi],
        "HbA1c_level": [HbA1c_level],
        "blood_glucose_level": [blood_glucose_level],
        "smoking_history_former": [smoking_history_former],
    }
)

# ---------- PREDICT BUTTON ----------
predict_clicked = st.button(
    "🩻 Predict Diabetes Risk", type="primary", use_container_width=True
)

if predict_clicked:
    if not patient_name:
        st.error("Please enter the patient name before predicting.")
    else:
        with st.spinner("Analyzing patient data..."):
            pred_label = xgb_model.predict(input_data)[0]
            if hasattr(xgb_model, "predict_proba"):
                proba = xgb_model.predict_proba(input_data)[0].max()
                prob_str = f"{proba:.1%}"
            else:
                prob_str = "N/A"

            result_str = "Diabetic" if pred_label == 1 else "Non-Diabetic"

            record = {
                "name": patient_name,
                "result": result_str,
                "age": age,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "probability": prob_str,
            }

            # Update session state
            st.session_state["history"].append(record)
            st.session_state["current_prediction"] = record

# ---------- SHOW LATEST PREDICTION UNDER BUTTON ----------
if st.session_state["current_prediction"] is not None:
    rec = st.session_state["current_prediction"]
    st.markdown(
        f"""
        <div class="prediction-card">
            <h2>{'✅ NON-DIABETIC' if rec['result']=='Non-Diabetic' else '⚠️ DIABETIC'}</h2>
            <p><b>Patient:</b> {rec['name']}</p>
            <p><b>Result:</b> {rec['result']}</p>
            <p><b>Model Confidence:</b> {rec['probability']}</p>
            <p><b>Age:</b> {rec['age']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- HOW TO USE ----------
with st.expander("ℹ️ How to use this app"):
    st.write(
        "1. Enter patient name in the sidebar.\n"
        "2. Fill all health details in the main form.\n"
        "3. Click Predict Diabetes Risk.\n"
        "4. Latest result appears under the button; past results appear in the sidebar history."
    )
