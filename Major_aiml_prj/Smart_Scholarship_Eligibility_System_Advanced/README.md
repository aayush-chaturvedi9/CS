# Smart Scholarship Eligibility System — Advanced Edition

A premium Streamlit AI/ML training project featuring classification, regression, feature engineering, model comparison, cross-validation, GridSearchCV, confusion matrix, permutation feature importance and subgroup error review.

## Data warning
The included dataset is **synthetic demonstration data**. It does not represent official scholarship eligibility criteria or real scholarship awards. The model is for learning only.

## Windows / VS Code setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
If `python` is unavailable, try `py`.

## Run
```bash
python generate_dataset.py
python train_models.py
streamlit run app.py
```
Open the local URL printed by Streamlit (usually http://localhost:8501).

## Structure
- `generate_dataset.py` — creates reproducible synthetic sample data
- `ml_pipeline.py` — feature engineering and preprocessing
- `train_models.py` — model comparison, tuning, evaluation and saved artifacts
- `app.py` — premium interactive dashboard
- `models/` — trained classifier and regressor
- `reports/` — evaluation and explainability CSV/TXT files

## Models
Classification: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, tuned Random Forest.
Regression: Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor.

## Responsible adaptation
For real use, replace synthetic data with lawful, representative, verified data; map the exact official scheme criteria; review privacy and fairness; validate with domain experts; and retain human review.
