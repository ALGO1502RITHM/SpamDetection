import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import pickle

# load data
spam_data = pd.read_csv('anti-spam system/spam_detection_dataset.csv')
# print(spam_data.head())

# Dataset info and description
# print(spam_data.info())
# print(spam_data.describe())
# print(spam_data['is_spam'].value_counts())

# Preprocessing
scaler = StandardScaler()
scaled_features = scaler.fit_transform(spam_data[['num_links', 'num_words']])

# create dataframe for scaled features
scaled_spam_data = pd.DataFrame(scaled_features, columns=['num_links_scaled', 'num_words_scaled'])
# print(scaled_spam_data)

# concantenate scaled features to original features 
spam_data_scaled = pd.concat([scaled_spam_data, spam_data.drop(columns=['num_links', 'num_words'])], axis=1)

# Normalising sender_score
min_max_scaler = MinMaxScaler()
spam_data_scaled['sender_score'] = min_max_scaler.fit_transform(spam_data_scaled[['sender_score']])

# print(spam_data_scaled.head())

# Feature engineering
spam_data_scaled['links_per_word'] = spam_data_scaled['num_links_scaled'] / spam_data_scaled['num_words_scaled']

# Features and Targets
features_1 = ['num_links_scaled', 'num_words_scaled', 'has_offer', 'sender_score', 'all_caps', 'links_per_word']
target = ['is_spam']

X = spam_data_scaled[features_1]
# print(X)

y = spam_data_scaled[target]
# print(y)

# Split datasets into train, test, split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Applying SMOTE
smote = SMOTE(random_state=42)
X_train_smote,  y_train_smote = smote.fit_resample(X_train, y_train)

# Hyperparameter Tuning
param = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced']
}

# Randomforest
rf_model = GridSearchCV(RandomForestClassifier(random_state=42), param, cv=5, n_jobs=-1)
rf_model.fit(X_train, y_train)

# rf_model = RandomForestClassifier(random_state=42, n_jobs=-1,class_weight='balanced', n_estimators=100)
# rf_model.fit(X_train, y_train)

# Model Evaluation
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Visualizaton Precision-Recall for Randomforest
precision, recall, threshold = precision_recall_curve(y_test, y_proba_rf)
plt.plot(threshold, precision[:-1], label='Precision')
plt.plot(threshold, recall[:-1], label='Recall')
plt.xlabel('Threshold')
plt.legend()
# plt.show()

new_threshold = 0.76
y_pred_rf = (y_proba_rf >= new_threshold).astype(int)

# print("Best RandomForest Parameter:", grid_rf.best_params_)
# print("Best Parameter score:", grid_rf.best_score_)
# print(classification_report(y_test, y_pred_rf))
# print(confusion_matrix(y_test, y_pred_rf))

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train.values.ravel())

# Evaluate XGBoost model
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Visualization of Precision-Recalll for XGBoost
precision_xgb, recall_xgb, threshold_xgb = precision_recall_curve(y_test, y_proba_xgb)
plt.plot(threshold_xgb, precision_xgb[:-1], label='Precision_xgb')
plt.plot(threshold_xgb, recall_xgb[:-1], label='Recall_xgb')
plt.xlabel('Threshold')
plt.legend()
plt.show()

new_threshold_1 = 0.42
y_pred_xgb = (y_proba_xgb >= new_threshold_1).astype(int)

print(classification_report(y_test, y_pred_xgb))

pickle.dump(xgb_model, open("model.pkl", "wb"))