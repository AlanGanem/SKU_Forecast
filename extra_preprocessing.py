# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 18:42:09 2019

@author: ganem
"""

import numpy as np
import pandas as pd
import itertools
import TimeSeriesUtils as TSU
import sklearn


#read data and convert to np array


dummy_columns = ['Agency','SKU','Month','Quarter']
prefixes = ['','','Month_','Quarter_']
dummy_features = {column:sorted([feature for feature in merged_data[column].unique()]) for column in dummy_columns}


agency_dummies = sorted([agency for agency in merged_data['Agency'].unique()])
sku_dummies = sorted([sku for sku in merged_data['SKU'].unique()])
month_dummies = sorted(['Month_'+str(month) for month in merged_data['Month'].unique()])
quarter_dummies = sorted(['Quarter_'+str(quarter) for quarter in merged_data['Quarter'].unique()])


categorical_future_vars = ['Easter Day', 'Good Friday', 'New Year', 'Christmas',
       'Labor Day', 'Independence Day', 'Revolution Day Memorial',
       'Regional Games ', 'FIFA U-17 World Cup', 'Football Gold Cup',
       'Beer Capital', 'Music Fest'] + agency_dummies + sku_dummies + month_dummies + weekday_columns
dependent_variable = ['Volume']
continuous_future_vars = ['Avg_Max_Temp','Industry_Volume','Soda_Volume','Avg_Population_2017', 'Avg_Yearly_Household_Income_2017','Discount','Local_Relative_Temp']
past_vars = ['Industry_Volume','Soda_Volume','Volume']

encoder_inputs = past_vars
decoder_inputs = categorical_future_vars + continuous_future_vars + dependent_variable


merged_data_dummies = pd.get_dummies(merged_data,columns = dummy_columns, prefix = prefixes,prefix_sep = '')
features = [feature for feature in merged_data_dummies.columns if feature not in  ['Agency','SKU','Volume']]+['Volume']

rng = pd.DataFrame(index = list(merged_data.index.unique()), columns = merged_data.columns)
combs = set(list(map(tuple,merged_data[['Agency','SKU']].values)))

merged_data_dummies,scaler = TSU.scale_df(merged_data_dummies,sklearn.preprocessing.MinMaxScaler())
merged_data_dummies = merged_data_dummies[features]
train_data_dict = {
        agency+' '+sku: merged_data_dummies[(merged_data_dummies[agency] == 1)&(merged_data_dummies[sku] == 1)][features].fillna(method = 'ffill', limit= 3).fillna(0)[features].sort_index()
        for agency,sku in combs}

for key in train_data_dict.keys():
    if np.all(np.isnan(train_data_dict[key].values)):
        train_data_dict.pop(key, None)

def data_transformer(train_data_dict, key, pred_period, look_back_period, encoder_inputs, decoder_inputs, **kwargs):
    
    X_train, y_train, X_val, y_val = TSU.chunk_and_concatenate_dict({key:train_data_dict[key].assign(date = train_data_dict[key].index)},pred_period,look_back_period,encoder_inputs,['date']+decoder_inputs,**kwargs)
    X_covars_train = y_train.take(range(1,y_train.shape[-1]-1),axis = -1)
    X_covars_val = y_val.take(range(1,y_val.shape[-1]-1),axis = -1)
    period_train = y_train.take([0],axis = -1)
    period_val = y_val.take([0],axis = -1)
    y_train = y_train.take([-1],axis = -1)
    y_val = y_val.take([-1],axis = -1)
        
    return {'period_train':period_train,'period_val':period_val,'X_train':X_train, 'X_covars_train':X_covars_train, 'y_train':y_train, 'X_val':X_val, 'X_covars_val':X_covars_val, 'y_val':y_val}



i = 0
i+=1
plt.clf()
train_data_dict[list(train_data_dict.keys())[i]]['Volume'].plot()
key = 'Agency_05 SKU_01'