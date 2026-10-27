"""Streamlit web application for Student Performance Prediction."""
from pathlib import Path
import pandas as pd
import streamlit as st
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
FEATURES = ["study_hours", "attendance_percent", "previous_marks", "assignment_score"]
MODEL_FILES = {
    "Linear Regression": MODEL_DIR / "linear_regression.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
}

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")

@st.cache_resource
def load_models():
    loaded = {}
    for name, path in MODEL_FILES.items():
        if path.exists():
            loaded[name] = joblib.load(path)
    return loaded

st.title("🎓 Student Performance Prediction")
st.caption("A machine-learning demonstration comparing Linear Regression and Random Forest Regression.")

models = load_models()
if not models:
    st.error("Trained models were not found.")
    st.markdown("Open a terminal in the project folder and run `python train_model.py`, then refresh this page.")
    st.stop()

with st.sidebar:
    st.header("Student inputs")
    study_hours = st.slider("Study hours per day", 0.0, 16.0, 3.0, 0.5)
    attendance = st.slider("Attendance (%)", 0, 100, 85)
    previous = st.slider("Previous exam marks (%)", 0, 100, 70)
    assignment = st.slider("Assignment score (%)", 0, 100, 75)
    selected = st.selectbox("Prediction model", ["Compare both algorithms"] + list(models.keys()))
    predict_clicked = st.button("Predict marks", type="primary", use_container_width=True)

st.subheader("How the system works")
st.write("Enter the academic indicators in the sidebar. The app sends them to the trained regression model and displays estimated final marks. These are model estimates, not guaranteed outcomes.")

input_df = pd.DataFrame([{
    "study_hours": study_hours,
    "attendance_percent": attendance,
    "previous_marks": previous,
    "assignment_score": assignment,
}], columns=FEATURES)

if predict_clicked:
    names = list(models) if selected == "Compare both algorithms" else [selected]
    results = []
    for name in names:
        value = float(models[name].predict(input_df)[0])
        value = max(0.0, min(100.0, value))
        results.append({"Model": name, "Estimated final marks": round(value, 1)})
    st.subheader("Prediction results")
    cols = st.columns(len(results))
    for col, result in zip(cols, results):
        with col:
            st.metric(result["Model"], f'{result["Estimated final marks"]:.1f} / 100')
    if len(results) == 2:
        difference = abs(results[0]["Estimated final marks"] - results[1]["Estimated final marks"])
        st.info(f"The two model estimates differ by {difference:.1f} marks. This difference is not a measure of which prediction is correct.")
else:
    st.info("Choose the inputs and click **Predict marks** to view estimates.")

st.divider()
st.subheader("Model evaluation")
metrics_path = MODEL_DIR / "evaluation_results.csv"
if metrics_path.exists():
    metrics = pd.read_csv(metrics_path)
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.caption("MAE/RMSE: lower is better. R²: higher is generally better. Metrics are calculated on a held-out test split.")
else:
    st.warning("Evaluation results not found. Run train_model.py first.")

with st.expander("Dataset information and limitations"):
    st.markdown("""
    - Expected columns: `study_hours`, `attendance_percent`, `previous_marks`, `assignment_score`, `final_marks`.
    - If no CSV exists, the training script creates 300 synthetic rows.
    - Synthetic data is useful for demonstrating code, but it is not evidence of real-world performance.
    - Use an appropriate real dataset and explain any differences in feature definitions before making research claims.
    - Avoid storing names, student IDs, or other identifying information.
    - Do not use this prototype for high-stakes decisions about students.
    """)
