# ==========================================================
# GUIDE DATASET SOC OPTIMIZATION PIPELINE
# research on optimizing SOC through Big Data analysis 
# ==========================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ==========================================================
# 1. LOAD DATA
# ==========================================================

DATA_PATH = "/home/ali/guidereduced2ndver.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.strip()

print("Dataset Shape:", df.shape)
print(df['IncidentGrade'].value_counts())

# ==========================================================
# 2. CREATE BINARY LABEL
# ==========================================================

df['label'] = df['IncidentGrade'].apply(
    lambda x: 0 if x in ['BenignPositive', 'FalsePositive'] else 1
)

print("\nLabel Distribution:")
print(df['label'].value_counts(normalize=True))

# ==========================================================
# 3. REMOVE LEAKAGE COLUMNS
# ==========================================================

leakage_cols = [
    'IncidentGrade',
    'LastVerdict',
    'ActionGrouped',
    'ActionGranular'
]

drop_cols = [
    'Id','AlertId','IncidentId',
    'Sha256','Url','RegistryKey',
    'RegistryValueName','RegistryValueData',
    'FolderPath','AccountSid','AccountUpn',
    'AccountObjectId','NetworkMessageId'
]

df = df.drop(columns=[c for c in leakage_cols + drop_cols if c in df.columns])

# ==========================================================
# 4. FEATURE ENGINEERING
# ==========================================================

# ---- Timestamp Features ----
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['hour'] = df['Timestamp'].dt.hour
    df['dayofweek'] = df['Timestamp'].dt.dayofweek
    df = df.drop(columns=['Timestamp'])

# ---- MITRE Count ----
if 'MitreTechniques' in df.columns:
    df['MitreCount'] = df['MitreTechniques'].fillna('').apply(
        lambda x: len(str(x).split(';')) if x != '' else 0
    )
    df = df.drop(columns=['MitreTechniques'])

# ==========================================================
# 5. SPLIT FEATURES AND LABEL
# ==========================================================

X = df.drop('label', axis=1)
y = df['label']

# ==========================================================
# 6. ENCODE CATEGORICAL FEATURES (ONCE)
# ==========================================================

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print("Encoded Feature Shape:", X.shape)

==========================================================
# 6.5 EXPORT OF PREPROCESSED DATASET
# ==========================================================

print("\nDo you want to save the preprocessed & feature engineered dataset?")
user_choice = input("Type YES to save or NO to skip: ").strip().lower()

if user_choice == "yes":

    # Combine features and label for export
    processed_dataset = pd.concat([X, y], axis=1)

    save_path = input(
        "Enter full path where CSV should be saved (e.g., /home/ali/processed_GUIDE.csv): "
    ).strip()

    processed_dataset.to_csv(save_path, index=False)

    print("\nPreprocessed dataset saved successfully at:")
    print(save_path)

else:
    print("\nSkipping dataset export and continuing pipeline...")

# ==========================================================
# 7. TRAIN-TEST SPLIT
# ==========================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.15,
    stratify=y,
    random_state=42
)

# ==========================================================
# 8. RANDOM FOREST
# ==========================================================

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_probs = rf.predict_proba(X_test)[:,1]

# ==========================================================
# 9. XGBOOST
# ==========================================================

import xgboost as xgb

scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_probs = xgb_model.predict_proba(X_test)[:,1]

# ==========================================================
# 10. ISOLATION FOREST (ANOMALY)
# ==========================================================

# Determine minority class proportion
minority_ratio = min(y_train.mean(), 1 - y_train.mean())

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

iso = IsolationForest(
    contamination=minority_ratio,
    random_state=42,
    n_jobs=-1
)

iso.fit(X_train)

iso_scores_raw = iso.decision_function(X_test)

scaler = MinMaxScaler()
iso_scores = scaler.fit_transform(
    iso_scores_raw.reshape(-1,1)
).flatten()

# ==========================================================
# 11. FUSION MODEL
# ==========================================================

alpha, beta, gamma = 0.5, 0.3, 0.2

risk_score = alpha*xgb_probs + beta*iso_scores + gamma*rf_probs

threshold = 0.5
final_preds = (risk_score >= threshold).astype(int)

# ==========================================================
# 12. EVALUATION
# ==========================================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, final_preds)
TN, FP, FN, TP = cm.ravel()

precision = TP / (TP + FP)
recall = TP / (TP + FN)
fpr = FP / (FP + TN)
f1 = 2*(precision*recall)/(precision+recall)

print("\nConfusion Matrix:\n", cm)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("FPR:", fpr)

# ==========================================================
# 13. FEATURE IMPORTANCE
# ==========================================================

import matplotlib.pyplot as plt

importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X_train.columns)
top_features = feat_imp.sort_values(ascending=False).head(15)

print("\nTop 15 Features:")
print(top_features)



#===================================================
# 13.5 GRAPHICAL CONFUSION MATRIX AND FEATURE IMPORTANCE
#=========================================================

print('\n visual confusion matrix')

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("GUIDE Dataset Confusion Matrix")
plt.show()

print('\n feature importance')
import pandas as pd
import matplotlib.pyplot as plt

importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X_train.columns)

top_features = feat_imp.sort_values(ascending=False).head(15)

print("\nTop 15 Important Features:")
print(top_features)

# Optional visualization
plt.figure(figsize=(8,6))
top_features.sort_values().plot(kind='barh')
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.show()
# ==========================================================
# 14. SHUFFLE SANITY TEST
# ==========================================================

from sklearn.metrics import accuracy_score

y_train_shuffled = y_train.sample(frac=1, random_state=42)

rf_shuffle = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_shuffle.fit(X_train, y_train_shuffled)
shuffle_preds = rf_shuffle.predict(X_test)

shuffle_accuracy = accuracy_score(y_test, shuffle_preds)

print("\nShuffle Test Accuracy:", shuffle_accuracy)

# ==========================================================
# END OF PIPELINE
# ==========================================================
