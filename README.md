# Customer Churn Prediction

Churn occurs when customers stop using a service. This directly impacts a company’s revenue. Accurately predicting churn allows businesses to take proactive retention measures.

This project uses a Random Forest classifier to predict customer churn using the Telco Customer Churn dataset. It includes preprocessing, class balancing, hyperparameter tuning, and overfitting detection.

## Dataset
The dataset is from IBM's Telco Customer Churn data and includes 21 features related to customer demographics, account information, and service usage.

## Features
- Categorical variables: Gender, Contract type, Payment method, etc.
- Numerical variables: Monthly charges, Total charges, Tenure, etc.
- Target variable: Churn (Yes/No)

## Workflow
1. Load and clean data
2. Impute missing values and encode features
3. Balance dataset using SMOTEENN
4. Train baseline Random Forest model
5. Perform hyperparameter tuning with RandomizedSearchCV
6. Evaluate model using F1-score and AUC-ROC
7. Visualize learning curve and confusion matrix
8. Check for overfitting using train-test metric gap

## Performance
- **F1-score:** ~0.97
- **AUC-ROC:** ~0.99
- No significant overfitting observed

## Tools & Libraries
- Python, Pandas, NumPy, Matplotlib
- Scikit-learn
- imbalanced-learn
