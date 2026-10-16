from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "livestock_symptoms.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "livestock_model.joblib"
REPORT_PATH = MODEL_DIR / "evaluation_report.txt"

SYMPTOMS = [
    "fever", "reduced_appetite", "coughing", "nasal_discharge",
    "diarrhea", "bloating", "lameness", "mouth_blisters",
    "udder_swelling", "abnormal_milk", "weight_loss", "weakness"
]
TARGET = "disease"
ANIMAL_COL = "animal_type"
FEATURES = [ANIMAL_COL] + SYMPTOMS

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = set(FEATURES + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df = df[FEATURES + [TARGET]].dropna()

    for col in SYMPTOMS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
        if not df[col].isin([0, 1]).all():
            raise ValueError(f"Column '{col}' must contain only 0 or 1.")
    if df[TARGET].nunique() < 2:
        raise ValueError("At least two different disease labels are required.")
    if df[TARGET].value_counts().min() < 2:
        raise ValueError("Each disease label needs at least two rows for a train/test split.")

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("animal", OneHotEncoder(handle_unknown="ignore"), [ANIMAL_COL]),
            ("symptoms", "passthrough", SYMPTOMS),
        ],
        remainder="drop"
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ))
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    report = (
        "DEMO EVALUATION REPORT\n"
        "IMPORTANT: bundled dataset is synthetic and not clinically valid.\n"
        "Metrics only demonstrate the evaluation workflow; they do not measure real-world veterinary accuracy.\n\n"
        f"Rows used: {len(df)}\n"
        f"Training rows: {len(X_train)}\n"
        f"Test rows: {len(X_test)}\n"
        f"Accuracy on held-out synthetic rows: {accuracy_score(y_test, predictions):.3f}\n\n"
        + classification_report(y_test, predictions, zero_division=0)
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    joblib.dump({"model": pipeline, "features": FEATURES}, MODEL_PATH)

    print(report)
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved evaluation report: {REPORT_PATH}")
    print("\nReminder: replace synthetic data with veterinarian-verified data before drawing conclusions.")

if __name__ == "__main__":
    main()
