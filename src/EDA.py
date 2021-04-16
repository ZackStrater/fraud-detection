import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


df = pd.read_csv('../data/data_cleaned.csv')
df.drop('org_twitter', axis=1, inplace=True)

print(df.groupby('acct_type').mean())
