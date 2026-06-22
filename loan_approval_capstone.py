import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

"""DataSet"""

import kagglehub

path = kagglehub.dataset_download("jumpingdino/german-credit-dataset")
print("Path to dataset files:", path)

print(os.listdir(path))

"""Data Preparation"""

df = pd.read_csv(os.path.join(path, "german_credit_data.csv"))

print(df.shape)

print(df.describe())

print(df.head())

print(df.columns.tolist())

print(df['target'].value_counts())

print(df[['status_and_sex', 'secondary_obligor', 'n_guarantors']].nunique())
print(df['status_and_sex'].unique())
print(df['secondary_obligor'].unique())
print(df['n_guarantors'].unique())

print(df.groupby('n_guarantors')['age'].mean())
print(df.groupby('n_guarantors')['status_and_sex'].value_counts())

print(df.head())

"""Converting Good and Bad into 1 and 0"""

# Encode target: good = 1, bad = 0
df['target'] = df['target'].map({'good': 1, 'bad': 0})

print(df.groupby('status_account')['target'].mean())
print(df.groupby('status_and_sex')['target'].mean())
print(df.groupby('secondary_obligor')['target'].mean())

remaining_cols = ['purpose', 'status_savings', 'years_employment',
                   'credit_history', 'collateral', 'housing', 'job',
                   'other_installment_plans', 'telephone', 'is_foreign_worker']

for col in remaining_cols:
    print(f"\n{col}:")
    print(df.groupby(col)['target'].mean().sort_values(ascending=False))

print(pd.crosstab(df['status_account'], df['credit_history'], values=df['target'], aggfunc='mean'))

# Does the foreign worker gap persist within each status_account group?
print(pd.crosstab(df['status_account'], df['is_foreign_worker'],
            values=df['target'], aggfunc='mean'))

# Same, but for credit_history
print(pd.crosstab(df['credit_history'], df['is_foreign_worker'],
            values=df['target'], aggfunc='mean'))

# Check sample size first - this matters a lot here
print(df['is_foreign_worker'].value_counts())

# Summary stats split by approval outcome
print(df.groupby('target')[['credit_amount', 'month_duration', 'age']].describe())

print(df.groupby('target')['credit_amount'].median())
print(df.groupby('target')['month_duration'].median())

print(df.groupby('target')['age'].median())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col in zip(axes, ['credit_amount', 'month_duration', 'age']):
    df.boxplot(column=col, by='target', ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.show()

print(df[['credit_amount', 'month_duration', 'age']].corr())

print(df.isnull().sum())

"""Imputation: there's nothing to impute. No missing data

Separate features and target
"""

X = df.drop(columns=['target'])
y = df['target']

"""Encode categorical variable"""

nominal_cols = ['purpose', 'housing', 'status_and_sex', 'secondary_obligor',
                 'job', 'other_installment_plans', 'collateral',
                 'telephone', 'is_foreign_worker', 'status_account',
                 'status_savings', 'years_employment', 'credit_history']

X_encoded = pd.get_dummies(X, columns=nominal_cols, drop_first=True)

"""Train-test split"""

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)
print(X_train.shape, X_test.shape)

print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

numeric_cols = ['credit_amount', 'month_duration', 'age', 'payment_to_income_ratio',
                 'residence_since', 'n_credits', 'n_guarantors']

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

"""Model Selection"""

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    print(f"{name} trained.")

"""Evaluation"""

results = {}

for name, model in models.items():
    preds = model.predict(X_test_scaled)   # <- always scaled, no branching
    results[name] = {
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1-Score': f1_score(y_test, preds)
    }

results_df = pd.DataFrame(results).T
print(results_df)

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, (name, model) in zip(axes, models.items()):
    preds = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm, display_labels=['Bad (0)', 'Good (1)']).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.show()

for name, model in models.items():
    preds = model.predict(X_test_scaled)
    print(f"\n{name}")
    print(classification_report(y_test, preds))

print("\nCost-weighted comparison (5x cost for approving a bad-credit applicant,")
print("1x cost for rejecting a good-credit applicant):")
for name, model in models.items():
    preds = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, preds)
    # cm layout: rows/cols ordered [0, 1] -> [[TN, FP], [FN, TP]]
    bad_approved = cm[0][1]   # actually Bad(0), predicted Good(1)
    good_rejected = cm[1][0]  # actually Good(1), predicted Bad(0)
    total_cost = 5 * bad_approved + 1 * good_rejected
    print(f"{name}: bad_approved={bad_approved}, good_rejected={good_rejected}, cost={total_cost}")
