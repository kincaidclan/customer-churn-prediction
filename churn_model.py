# Re-import necessary libraries after kernel reset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report, roc_curve, ConfusionMatrixDisplay
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Clean data
df = df.dropna(subset=['TotalCharges'])
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
y = df['Churn'].map({'No': 0, 'Yes': 1})
X = df.drop(['customerID', 'Churn'], axis=1)

# Define column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()


# Imputer + scaler for numeric columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Imputer + one-hot encoder for categorical columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Resampling
smote_enn = SMOTEENN(random_state=42)
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('sampler', smote_enn)
])

# Apply transformations
X_resampled, y_resampled = pipeline.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Initialize model
rf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Baseline evaluation
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]
baseline_f1 = f1_score(y_test, y_pred)
baseline_auc = roc_auc_score(y_test, y_proba)
baseline_report = classification_report(y_test, y_pred)

# Hyperparameter tuning
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt'],
    'bootstrap': [False]
}
rs = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,
    scoring='f1',
    cv=5,
    return_train_score=True,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
rs.fit(X_train, y_train)
best_rf = rs.best_estimator_
print("Best hyperparameters:", rs.best_params_)

y_pred_tuned = best_rf.predict(X_test)
y_proba_tuned = best_rf.predict_proba(X_test)[:, 1]

# Evaluate tuned model
tuned_f1 = f1_score(y_test, y_pred_tuned)
tuned_auc = roc_auc_score(y_test, y_proba_tuned)
tuned_report = classification_report(y_test, y_pred_tuned)

# Overfitting check
y_train_pred = best_rf.predict(X_train)
y_train_proba = best_rf.predict_proba(X_train)[:, 1]
train_f1 = f1_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_proba)
gap_f1 = train_f1 - tuned_f1
gap_auc = train_auc - tuned_auc
overfitting_status = {
    "F1-score gap": (gap_f1, "OVERFITTING" if gap_f1 > 0.05 else "No overfitting"),
    "AUC-ROC gap": (gap_auc, "OVERFITTING" if gap_auc > 0.05 else "No overfitting")
}
print(overfitting_status)
print("\nTuned Random Forest Performance")
print(f"F1-score : {f1_score(y_test, y_pred_tuned):.3f}")
print(f"AUC-ROC  : {roc_auc_score(y_test, y_proba_tuned):.3f}\n")
print(classification_report(y_test, y_pred_tuned))

