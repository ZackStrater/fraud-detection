
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

scaler = MinMaxScaler()

df = pd.read_csv('../data/data_cleaned.csv')
df.drop('org_twitter', axis=1, inplace=True)

target_mask = df['acct_type'] == 0

df_normal = df[target_mask]
df_normal.pop('acct_type')
df_fraud = df[~target_mask]
df_fraud.pop('acct_type')

scaler.fit(df_normal)
df_fraud = scaler.transform(df_fraud)
df_normal = scaler.transform(df_normal)

print(df_normal.shape)


x_normal_train, x_normal_test = train_test_split(df_normal, test_size=0.25, random_state=42)


# model = Sequential()
# model.add(Dense(42, input_dim=x_normal_train.shape[1], activation='relu'))
# model.add(Dense(5, activation='relu')) # size to compress to
# model.add(Dense(42, activation='relu'))
# model.add(Dense(x_normal_train.shape[1])) # Multiple output neurons
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(x_normal_train, x_normal_train,verbose=1,epochs=100)
# model.save('skinny_mid')
model = load_model('thick_mid')

pred1 = model.predict(x_normal_test)
score1 = np.sqrt(metrics.mean_squared_error(pred1,x_normal_test))
pred2 = model.predict(x_normal_train)
score2 = np.sqrt(metrics.mean_squared_error(pred2,x_normal_train))
pred3 = model.predict(df_fraud)
score3 = np.sqrt(metrics.mean_squared_error(pred3,df_fraud))
print(f"Out of Sample Normal Score (RMSE): {score1}")
print(f"Insample Normal Score (RMSE): {score2}")
print(f"Fraud Score (RMSE): {score3}")

test_error_arrays = (x_normal_test - pred1)**2
test_error = np.sum(test_error_arrays, axis=1)


fraud_error_arrays = (df_fraud - pred3)**2
fraud_error = np.sum(fraud_error_arrays, axis=1)


test_mse = np.mean(test_error)
fraud_mse = np.mean(fraud_error)

difference = (test_mse-fraud_mse)
threshold = (test_mse + fraud_mse)/10

print(threshold)
print(test_mse)
print(fraud_mse)
print('fraud error')
print(fraud_error[0:100])
print('test error'[0:100])
print(test_error)

test_hard_pred = np.where(fraud_error > threshold, 1, 0)
fraud_hard_pred = np.where(test_error < threshold, 1, 0)
print(np.mean(test_hard_pred))
print(np.mean(fraud_hard_pred))



# recall higher (TPR)
# specificity higher (TNR)
# FP rate = FP/N
# precision = TP/TP + FP
# precision when you make a prediciton how much of the time is it right