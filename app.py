from pathlib import Path
import pandas as pd
import streamlit as st
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "livestock_model.joblib"
SYMPTOMS = [
    "fever", "reduced_appetite", "coughing", "nasal_discharge",
    "diarrhea", "bloating", "lameness", "mouth_blisters",
    "udder_swelling", "abnormal_milk", "weight_loss", "weakness"
]
ANIMALS = ["Cow", "Buffalo", "Goat", "Sheep"]
LABELS = {
    "fever": "Fever",
    "reduced_appetite": "Reduced appetite",
    "coughing": "Coughing",
    "nasal_discharge": "Nasal discharge",
    "diarrhea": "Diarrhea",
    "bloating": "Bloating / swollen abdomen",
    "lameness": "Lameness / difficulty walking",
    "mouth_blisters": "Mouth blisters or sores",
    "udder_swelling": "Udder swelling",
    "abnormal_milk": "Abnormal milk",
    "weight_loss": "Weight loss",
    "weakness": "Weakness"
}

st.set_page_config(page_title="Livestock Disease Prediction", page_icon="🐄", layout="wide")
st.title("🐄 Animal Disease Prediction")
st.subheader("Machine Learning and Symptom Analysis")
st.write("College project demonstration for cattle, buffaloes, goats, and sheep.")

st.warning(
    "DEMO ONLY: The included dataset is synthetic and its labels/patterns are illustrative, "
    "not veterinarian-validated. Predictions are not diagnoses and must not guide treatment. "
    "Use a veterinarian-verified dataset and consult a veterinarian for real animals."
)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python train_model.py` from the project folder first.")
    st.stop()

try:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    model_features = artifact["features"]
except Exception as exc:
    st.error(f"Could not load model: {exc}. Run `python train_model.py` again.")
    st.stop()

with st.form("symptom_form"):
    col1, col2 = st.columns(2)
    with col1:
        animal_type = st.selectbox("Animal type", ANIMALS)
    with col2:
        age_years = st.number_input("Approximate age (years)", min_value=0.0, max_value=40.0, value=3.0, step=0.5)
    st.markdown("### Select all symptoms observed")
    st.caption("Leave a symptom unchecked only if it has not been observed. If unsure, seek veterinary advice.")
    selected = {}
    symptom_cols = st.columns(3)
    for i, symptom in enumerate(SYMPTOMS):
        with symptom_cols[i % 3]:
            selected[symptom] = int(st.checkbox(LABELS[symptom], key=symptom))
    submitted = st.form_submit_button("Analyze symptoms", type="primary", use_container_width=True)

if submitted:
    row = {"animal_type": animal_type}
    row.update(selected)
    X_input = pd.DataFrame([row])
    # Ensure exact expected feature order.
    X_input = X_input[model_features]
    prediction = model.predict(X_input)[0]
    st.markdown("---")
    st.markdown("### Result")
    st.info(f"Model-predicted demo category: **{prediction}**")
    st.caption(
        "This is a model output from synthetic demonstration data, not a reliable estimate of disease "
        "probability. The model has not been validated for veterinary use."
    )

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)[0]
        probability_table = pd.DataFrame({
            "Demo class": model.named_steps["classifier"].classes_,
            "Model score (not clinical confidence)": probs
        }).sort_values("Model score (not clinical confidence)", ascending=False)
        st.markdown("#### Model scores")
        st.dataframe(probability_table, hide_index=True, use_container_width=True)

    if any(selected.values()):
        st.markdown("#### Symptoms entered")
        st.write(", ".join(LABELS[k] for k, v in selected.items() if v))
    else:
        st.warning("No symptoms were selected. The model still returns a class, but this output is especially uninformative.")

    st.markdown("#### Next steps")
    st.write(
        "- Contact a qualified veterinarian for examination and diagnosis.\n"
        "- Record when symptoms began and whether they are worsening.\n"
        "- Keep the animal comfortable and follow professional advice.\n"
        "- If the animal has severe breathing difficulty, collapses, cannot stand, has marked abdominal swelling, "
        "or a suspected contagious outbreak, seek urgent veterinary help and limit contact with other animals."
    )

with st.expander("About this project"):
    st.write(
        "The app uses a scikit-learn classification pipeline trained from a CSV file. "
        "The bundled CSV is synthetic and exists only to demonstrate the software workflow. "
        "Replace it with a reliable, veterinarian-reviewed dataset before any meaningful evaluation."
    )
    st.write("Age is collected for demonstration but is not used by this starter model.")
