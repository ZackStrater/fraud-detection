
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from IPython.display import display, HTML
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('../data/fixed_cleaned_data.csv')
df['org_twitter'].fillna(0, inplace=True)
y = df.pop('acct_type')
X = df

# test train split
X_gbtrain, X_gbtest, y_gbtrain, y_gbtest = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

labeled_train = X_gbtrain.copy()
labeled_train['acct_type'] = y_gbtrain
labeled_test = X_gbtest.copy()
labeled_test['acct_type'] = y_gbtest


train_target_mask = labeled_train['acct_type'] == 0
test_target_mask = labeled_test['acct_type'] == 0

# isolating just non-fraud for isolation forest and autoencoder
train_normal = labeled_train[train_target_mask]
train_fraud = labeled_train[~train_target_mask]
test_normal = labeled_test[test_target_mask]
test_fraud = labeled_test[~test_target_mask]



# XGboost model
# XGB = XGBClassifier()
# XGB.fit(X_gbtrain,y_gbtrain)
# dump(XGB, 'models/XGboost_model.joblib')


# Isolation Forest model
# IF = IsolationForest(n_estimators = 500, contamination = 0.17)
# IF.fit(train_normal)
# dump(IF, 'models/IsolationForest_model.joblib')


scaler = MinMaxScaler()
scaler.fit(train_normal)
auto_train_normal = scaler.transform(train_normal)
auto_train_fraud = scaler.transform(train_fraud)
auto_test_normal = scaler.transform(test_normal)
auto_test_fraud = scaler.transform(test_fraud)
dump(scaler, 'models/scaler.joblib')

# autoencoder skinny
# model = Sequential()
# model.add(Dense(43, input_dim=train_normal.shape[1], activation='relu'))
# model.add(Dense(5, activation='relu')) # size to compress to
# model.add(Dense(43, activation='relu'))
# model.add(Dense(train_normal.shape[1])) # Multiple output neurons
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(train_normal, train_normal,verbose=1,epochs=100)
# model.save('models/skinny_autoencoder')


# autoencoder thick
# model = Sequential()
# model.add(Dense(43, input_dim=train_normal.shape[1], activation='relu'))
# model.add(Dense(60, activation='relu')) # size to compress to
# model.add(Dense(43, activation='relu'))
# model.add(Dense(train_normal.shape[1])) # Multiple output neurons
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(train_normal, train_normal,verbose=1,epochs=100)
# model.save('models/thick_autoencoder')


def get_preds(formatted_data, if_model, xgb_model, data_scalar, skinny_auto_model, thick_auto_model):
    '''takes in correctly formatted data (#rows x 43 feature columns)
    returns dataframe of predictions from the models'''
    if_preds = if_model.predict(formatted_data)
    xgb_preds = xgb_model.predict_proba(formatted_data)
    scaled_data = data_scalar.transform(formatted_data)
    skinny_auto_model_arrays = skinny_auto_model.predict(scaled_data)
    skinny_errors =
    thick_auto_model_arrays = thick_auto_model.predict(scaled_data)
    thick_errors =

ensembled_RF = RandomForestClassifier()