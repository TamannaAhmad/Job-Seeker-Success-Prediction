import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('recruitment_data.csv')

# Basic Info
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Feature Types
numerical_features = [
    'Age', 'ExperienceYears', 'PreviousCompanies', 'DistanceFromCompany',
    'InterviewScore', 'SkillScore', 'PersonalityScore'
]

categorical_features = [
    'Gender', 'EducationLevel', 'RecruitmentStrategy'
]

target = 'HiringDecision'

# Preprocessing Pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Split data
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Models
models = {
    'Logistic Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
}

# Train & Evaluate Models
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    results[name] = {
        'accuracy': acc,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'model': model
    }
    
    print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
    print(classification_report(y_test, y_pred))

# Select best model
best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
best_model = results[best_model_name]['model']

print(f"\nBest model: {best_model_name}")

# Hyperparameter Tuning
print(f"\nHyperparameter tuning for {best_model_name}...")

if best_model_name == 'Logistic Regression':
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__solver': ['liblinear', 'saga']
    }
elif best_model_name == 'Random Forest':
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None]
    }
else:  # XGBoost
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.2]
    }

grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

tuned_model = grid_search.best_estimator_

# Final evaluation
y_pred = tuned_model.predict(X_test)
y_proba = tuned_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

print("\nTuned Model Results:")
print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

# Feature Importance
print("\nCalculating Feature Importance...")

# Get feature names after preprocessing
preprocessed_features = tuned_model.named_steps['preprocessor'].get_feature_names_out()

if best_model_name == 'Logistic Regression':
    importances = np.abs(tuned_model.named_steps['classifier'].coef_[0])
else:
    importances = tuned_model.named_steps['classifier'].feature_importances_

feature_importance = pd.DataFrame({
    'Feature': preprocessed_features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nTop Important Features:")
print(feature_importance)

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)

# Plot ROC Curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.close()

# Save model
joblib.dump(tuned_model, 'hiring_success_model.pkl')
print("\nModel and Feature Importance Saved!")

# Prediction Function
def predict_job_success(candidate_data, model_path='hiring_success_model.pkl'):
    """
    Predict hiring success probability for a new candidate.
    candidate_data: dict of input features
    """
    model = joblib.load(model_path)
    candidate_df = pd.DataFrame([candidate_data])
    probability = model.predict_proba(candidate_df)[0, 1]
    return probability

# Example
example_candidate = {
    'Age': 30,
    'Gender': 0,  # Male
    'EducationLevel': 3,  # Master's
    'ExperienceYears': 5,
    'PreviousCompanies': 2,
    'DistanceFromCompany': 10.5,
    'InterviewScore': 85,
    'SkillScore': 80,
    'PersonalityScore': 75,
    'RecruitmentStrategy': 2  # Moderate
}

success_prob = predict_job_success(example_candidate)
print(f"\nExample Candidate Hiring Probability: {success_prob:.4f}")
