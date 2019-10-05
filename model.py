# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 18:42:09 2019

@author: ganem
"""

import numpy as np
import pandas as pd
import itertools
import TimeSeriesUtils as TSU

latent_dim = 2
look_back_period = 8
pred_period = 3
dropout_rate = 0
batch_size = 128
#read data and convert to np array


agency_dummies = [agency for agency in merged_data['Agency'].unique()]
sku_dummies = [sku for sku in merged_data['SKU'].unique()]

categorical_future_vars = ['Easter Day', 'Good Friday', 'New Year', 'Christmas',
       'Labor Day', 'Independence Day', 'Revolution Day Memorial',
       'Regional Games ', 'FIFA U-17 World Cup', 'Football Gold Cup',
       'Beer Capital', 'Music Fest'] + agency_dummies + sku_dummies
dependent_variable = ['Volume']
continuous_future_vars = ['Avg_Max_Temp','Industry_Volume','Soda_Volume','Avg_Population_2017', 'Avg_Yearly_Household_Income_2017']
past_vars = ['Industry_Volume','Soda_Volume','Volume']

encoder_inputs = past_vars
decoder_inputs = categorical_future_vars + continuous_future_vars + dependent_variable

merged_data_dummies = pd.get_dummies(merged_data,columns = ['Agency','SKU'], prefix = '',prefix_sep = '')
features = [feature for feature in merged_data_dummies.columns if feature not in  ['Agency','SKU','Volume']]+['Volume']

rng = pd.DataFrame(index = list(merged_data.index.unique()), columns = merged_data.columns)
combs = set(list(map(tuple,merged_data[['Agency','SKU']].values)))

train_data_dict = {
        agency+' '+sku: merged_data_dummies[(merged_data_dummies[agency] == 1)&(merged_data_dummies[sku] == 1)][features].fillna(method = 'ffill', limit= 3).fillna(0)[features].sort_index()
        for agency,sku in combs}

for key in train_data_dict.keys():
    if np.all(np.isnan(train_data_dict[key].values)):
        train_data_dict.pop(key, None)

X_train, y_train, X_val, y_val = TSU.chunk_and_concatenate_dict(train_data_dict,pred_period,look_back_period,encoder_inputs,decoder_inputs)
X_covars_train = y_train.take(range(y_train.shape[-1]-1),axis = -1)
X_covars_val = y_val.take(range(y_val.shape[-1]-1),axis = -1)
y_train = y_train.take(-1,axis = -1)
y_val = y_val.take(-1,axis = -1)
