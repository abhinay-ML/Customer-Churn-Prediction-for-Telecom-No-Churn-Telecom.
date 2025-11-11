# Customer-Churn-Prediction-for-Telecom-No-Churn-Telecom.
Creating Churn risk scores that can be indicative to drive retention campaigns.
No Churn — Telecom Churn Prediction

Overview:
This project implements a churn-prediction pipeline for a telecom dataset (telecom_churn_data). Data is loaded from a MySQL database, cleaned and preprocessed, exploratory data analysis (EDA) is performed, features are engineered and encoded, and several classification models are trained and compared to predict whether a customer will churn. The project emphasizes model evaluation using ROC-AUC and classification reports and provides model explainability using SHAP.

Key steps performed in the notebook:
- Data ingestion: read data directly from a MySQL database using `pymysql` and `pandas`.
- Data cleaning: renamed columns for readability, converted numeric fields (removing commas), filled missing values, and handled zeros for area/usage fields where appropriate.
- Exploratory Data Analysis (EDA): churn class distribution, correlation heatmap for numerical features, histograms for key numeric columns, and churn vs categorical features (International Plan, VMail Plan, Area Code).
- Feature processing: created a binary `Churn_Flag` (1 for churn, 0 for no churn), encoded 'International Plan' and 'VMail Plan' as 0/1, and ensured 'Area Code' is integer. Dropped ID-like columns (Phone, State) before modeling.
- Train/test split: stratified 80/20 split for stable class representation.
- Models trained and evaluated:
  - Logistic Regression (class_weight='balanced') — baseline model, evaluated with ROC-AUC and classification report.
  - LightGBM classifier — strong tree-based model tuned for performance.
  - Random Forest classifier — ensemble baseline with feature importance and confusion matrix inspection.
- Model evaluation: ROC-AUC score, classification report (precision, recall, f1-score), and confusion matrices are printed for test set predictions.
- Explainability: SHAP used to compute global and detailed feature impacts for the best model (LightGBM) with summary plots for feature importance and per-feature effects.

Files:
- `no churn.ipynb` — main notebook containing all code (data ingestion, EDA, preprocessing, models, evaluation, SHAP).
- dataset: data is read from a MySQL table `telecom_churn_data` (the notebook connects to a DB with credentials). For local reproduction, export the table to CSV and place it in the repo.
- `README_no_churn.txt` — this file.

Quick start / How to run:
1. Ensure you have the dataset available locally (either export from MySQL or provide DB access). If using local CSV, update the notebook to read from the CSV instead of the DB.
2. Create and activate a Python virtual environment:
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
3. Install required libraries:
   pip install -r requirements.txt
   (example packages: pandas, numpy, matplotlib, seaborn, scikit-learn, lightgbm, shap, pymysql)
4. Open and run the notebook:
   jupyter notebook "no churn.ipynb"

Recommendations & Next Steps:
- Add data validation checks and explicit imputation strategies for columns with domain-specific meaning.
- Use one-hot encoding or target encoding for categorical variables with many levels if beneficial.
- Perform hyperparameter tuning (GridSearchCV / RandomizedSearchCV) for LightGBM and RandomForest; use cross-validation to obtain robust metrics.
- Consider resampling techniques (SMOTE, undersampling) if classes are highly imbalanced, or use class-weighted losses (already used for some models).
- Save the best model with `joblib.dump(best_model, 'no_churn_model.pkl')` for deployment and create a small inference script or REST endpoint.
- For production explainability, store SHAP values or integrate SHAP plots into a reporting notebook/dashboard.

License:
MIT — free to use and adapt for educational projects.

Notes:
- The notebook in this repo connects to a remote MySQL instance with credentials; for security, remove credentials before publishing and prefer environment variables or a local CSV for sharing the repo.
- If you want, I can generate a pinned `requirements.txt` for this notebook or export the trained model (`.pkl`) if you want a runnable artifact.
