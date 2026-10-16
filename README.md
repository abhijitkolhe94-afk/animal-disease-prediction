# Animal Disease Prediction Using Machine Learning and Symptom Analysis

A Streamlit college-project starter for livestock symptom classification using Python and scikit-learn.

## Important safety and dataset note

The included `data/livestock_symptoms.csv` is **synthetic demonstration data**. Its labels and symptom patterns are illustrative only and are not veterinarian-validated. The model must not be used to diagnose or treat real animals. Its scores and test metrics are not evidence of clinical accuracy.

For a meaningful academic evaluation, replace the sample CSV with a reliable, permissioned, veterinarian-reviewed dataset. Document the data source, species, label definitions, missing values, and limitations. Avoid making treatment recommendations from model predictions.

## Features

- Animal type selection: cow, buffalo, goat, sheep
- Checkbox symptom entry
- Random Forest classifier pipeline
- Train/test split and evaluation report
- Streamlit interface
- CSV validation and model artifact generation

## Folder structure

```text
animal_disease_prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── livestock_symptoms.csv
└── models/
    └── (generated after training)
```

## Setup

Python 3.10 or newer is recommended.

### Windows (PowerShell or VS Code terminal)

```powershell
cd animal_disease_prediction
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
cd animal_disease_prediction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

## Dataset format

The CSV must contain these columns:

- `animal_type`: Cow, Buffalo, Goat, or Sheep (or other documented species)
- Symptom columns listed below: use `1` for observed and `0` for not observed
- `disease`: target label supplied by a reliable, qualified source

Required symptom columns:
`fever`, `reduced_appetite`, `coughing`, `nasal_discharge`, `diarrhea`, `bloating`, `lameness`, `mouth_blisters`, `udder_swelling`, `abnormal_milk`, `weight_loss`, `weakness`

Example row (format illustration only, not a medical association):
```csv
animal_type,fever,reduced_appetite,coughing,nasal_discharge,diarrhea,bloating,lameness,mouth_blisters,udder_swelling,abnormal_milk,weight_loss,weakness,disease
Cow,0,1,0,0,0,0,0,0,0,0,1,1,Example label
```

Use multiple labeled examples per class. Ensure labels are assigned by a trustworthy veterinary source, not guessed from symptoms. Avoid train/test leakage (for example, duplicate records in both sets).

## Commands

Train model and create report:
```bash
python train_model.py
```

Run app:
```bash
streamlit run app.py
```

Generated files:
- `models/livestock_model.joblib`
- `models/evaluation_report.txt`

## Suggested guide demonstration

1. Explain the problem statement and intended users.
2. Show the CSV columns and discuss why reliable labels matter.
3. Run `python train_model.py` and explain train/test split and evaluation metrics.
4. Launch the Streamlit app and enter sample symptoms.
5. Explain that current predictions are demonstration outputs only because the bundled dataset is synthetic.
6. Describe next steps: source verified data, validate with veterinary experts, test generalization across species/farms, and document limitations.
