
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
scaler = MinMaxScaler()

df = pd.read_csv('path')
target_mask = df['acct_type'] == 0

df_train = df[target_mask]


