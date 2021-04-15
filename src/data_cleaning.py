import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from datetime import datetime as dt
import requests
from bs4 import BeautifulSoup
import re
import pickle


data = pd.read_json('data/data.json')

def drop_columns(df1, list_o_columns=['approx_payout_date',
 'sale_duration2',
 'gts',
 'num_payouts',
 'num_order',
 'sequence_number']):
    '''
    drops from df1 all columns not existing in df2
    '''
    for col in list_o_columns:
        if col in df1.columns:
            df1.drop(col, axis=1, inplace = True)
    return df1


def df_apply(df1, encoder_filepath):
    df = df1.copy()
    df['acct_type'] = df['acct_type'].apply(lambda x: 0 if x=='premium' else 1) # split fraud and not fraud
    df['venue_address'] = df['venue_address'].apply(lambda x: 0 if x=='' else 1) # helped classify missing addresses
    df['email_domain'] = df['email_domain'].apply(lambda x: 1 if x in ['gmail.com','yahoo.com','hotmail.com','aol.com','live.com'] else 0) # encode top 5 emails as 1 else 0
    df['user_created'] = df['user_created'].apply(dt.utcfromtimestamp) 
    df['event_created'] = df['event_created'].apply(dt.utcfromtimestamp)
    df['event_end'] = df['event_end'].apply(dt.utcfromtimestamp)
    df['event_published'] = df['event_published'].apply(lambda x: dt.utcfromtimestamp(x) if ~np.isnan(x) else x)
    df['event_start'] = df['event_start'].apply(lambda x: dt.utcfromtimestamp(x) if ~np.isnan(x) else x)
#     df['description'] = df['description'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
#     df['description'] = df['description'].apply(lambda x: x.replace('\n','').replace('\r','').replace('\xa0',''))
#     df['org_desc'] = df['org_desc'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
#     df['org_desc'] = df['org_desc'].apply(lambda x: x.replace('\n','').replace('\r','').replace('\xa0',''))    

    df['same_loc'] = df['country'] == df['venue_country']
    df['same_loc'] = df['same_loc'] * 1
    df.fillna({'venue_state': 'none', 'venue_country': 'none', 'country': 'none'}, inplace=True)
    def country_encode(x, prefix):
        if x == 'US':
            return prefix + 'US'
        elif x == 'GB':
            return prefix + 'GB'
        elif x == 'CA':
            return prefix + 'CA'
        elif x == 'none':
            return prefix + 'none'
        elif x == '':
            return prefix + ''
        else:
            return prefix + 'other'
    df['venue_country'] = df['venue_country'].apply(lambda x: country_encode(x, 'venue'))
    df['country'] = df['country'].apply(lambda x: country_encode(x, 'country'))
    df['listed'] = df['listed'].map({'y':1, 'n':0})
    df['delivery_method'].fillna(-1, inplace=True)
    df['has_header'].fillna(-1, inplace=True)
    df['org_facebook'].fillna(-1, inplace=True)
    df['sale_duration'].fillna(0, inplace=True)
    df['venue_latitude'].fillna(df['venue_latitude'].mean(), inplace=True)
    df['venue_longitude'].fillna(df['venue_longitude'].mean(), inplace=True)
    df['org_twitter'].fillna(0)
    df['org_facebook'].fillna(0)
    def get_ticket_info(lst):
        if len(lst) > 0:
            costs = []
            count = []
            for dic in lst:
                costs.append(dic['cost'])
                count.append(dic['quantity_total'])
            return pd.Series([max(costs), min(costs), max(count), min(count)])
        else:
            return pd.Series([0, 0, 0, 0])
    df[['max_cost', 'min_cost', 'max_tickets', 'min_tickets']]= df['ticket_types'].apply(lambda x: get_ticket_info(x))
    with open(encoder_filepath, 'rb') as f:
        fit_cat_encoder = pickle.load(f)
    cat_features = fit_cat_encoder.transform(df[['venue_country', 'country', 'currency', 'payout_type']])
    array_cat_features = cat_features.toarray()
    feature_labels = fit_cat_encoder.categories_
    feature_labels = np.concatenate(feature_labels, axis=0)
    for i, label in enumerate(feature_labels):
        df[label] = array_cat_features[:, i]
    return df
    

    
    
    
    
def drops(df, drop_columns = ['venue_latitude', 'venue_longitude', 'venue_name', 'description','event_created','event_end', 
                'event_published','event_start','name','object_id','org_desc','payee_name','org_name',
                'ticket_types','currency', 'venue_state','venue_country', 'venue_address', 'user_created',
               'payout_type', 'previous_payouts']):
    data = df.drop(drop_columns, axis = 1).copy()
    return data
    
    
    
def all(df, encoder_filepath):
    data = df.copy()
    data = drop_columns(data)
    data = df_apply(data, encoder_filepath)
    data = drops(data)
    return data
    
    
    
data = all(data, 'encoder.pickle')

print(data)
