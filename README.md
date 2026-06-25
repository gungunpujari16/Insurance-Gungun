# Insurance Claim Settlement Bias & Classification Streamlit Dashboard

This project is a ready-to-upload Streamlit dashboard for claim settlement analysis.
It is designed for the `Insurance.csv` structure with the target column `POLICY_STATUS`.

## What the dashboard covers

1. Descriptive analytics using cross-tabulation against policy status.
2. Diagnostic analysis for potential claim settlement bias by age, income, gender, zone/team, state, occupation, medical status, payment mode, and other segments.
3. Feature engineering before model training.
4. Supervised learning models:
   - KNN
   - Decision Tree
   - Random Forest
   - Gradient Boosting
5. Model comparison using:
   - Training accuracy
   - Testing accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC
   - ROC curve
   - Confusion matrix
   - Cross-validation stability
6. Automated findings and downloadable insight tables.

## Important privacy note

The actual `Insurance.csv` file is **not included** in this GitHub-ready folder because claimant/policy data can be sensitive. 
The app allows you to upload the CSV inside Streamlit after deployment.

A dummy `sample_data_template.csv` is included only to show the required column structure.

## Files included

```text
app.py
requirements.txt
README.md
.gitignore
.streamlit/config.toml
sample_data_template.csv
```

## Required columns

The dashboard works best when your CSV contains these columns:

```text
POLICY_NO
PI_NAME
PI_GENDER
SUM_ASSURED
ZONE
PAYMENT_MODE
EARLY_NON
PI_OCCUPATION
MEDICAL_NONMED
PI_STATE
REASON_FOR_CLAIM
PI_AGE
PI_ANNUAL_INCOME
POLICY_STATUS
```

The target column is:

```text
POLICY_STATUS
```

The dashboard treats values containing `Repudiate` as the positive/risk class.

## How to run locally

Open a terminal inside this folder and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then upload your actual `Insurance.csv` from the sidebar.

## How to upload to GitHub manually

1. Unzip this folder.
2. Create a new GitHub repository.
3. Upload all files and folders from this project folder.
4. Do **not** upload the real `Insurance.csv` unless your repository is private and your company permits it.
5. Commit the files.

## How to deploy on Streamlit Community Cloud

1. Go to Streamlit Community Cloud.
2. Click **New app**.
3. Select your GitHub repository.
4. Set the main file path as:

```text
app.py
```

5. Deploy the app.
6. Open the app and upload your actual `Insurance.csv` in the sidebar.

## Suggested interpretation standard

This dashboard does not automatically prove illegal or unethical bias. It highlights statistically unusual settlement differences that should be reviewed by claim managers, auditors, and compliance teams.

Use the dashboard to ask:

- Which groups have higher repudiation rates?
- Do differences remain after cross-feature segmentation?
- Are there team/zone/state patterns?
- Are income, age, medical status, payment mode, or early claim status interacting with claim outcome?
- Which model features are most influential in predicting repudiation?

## Troubleshooting

If deployment fails, check:

- `requirements.txt` is in the GitHub root folder.
- `app.py` is in the GitHub root folder.
- Streamlit main file path is exactly `app.py`.
- Your CSV has the expected target column `POLICY_STATUS`.
- `SUM_ASSURED` and `PI_ANNUAL_INCOME` can be text like `500,000`; the app cleans them automatically.
