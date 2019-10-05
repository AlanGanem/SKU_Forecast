# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:57:00 2019

@author: ganem
"""    
import calendar
from datetime import datetime

# get volume of previous periods
merged_data = merged_data.groupby(['SKU','Agency']).apply(lambda x: x.sort_index().assign(last_month = x[['Volume']].shift(1),last_2_month = x[['Volume']].shift(2),last_3_month = x[['Volume']].shift(3),last_year = x[['Volume']].shift(12))).reset_index(level = ['SKU','Agency'],drop = True)

merged_data = merged_data.fillna(method = 'bfill', limit = 12)

def hash_mapper(df):
    merged_data = df
    inv_hashmap = {hash(feature)%100000: feature for feature in set(merged_data[merged_data.columns[list(merged_data.dtypes == 'object')]].values.flatten())}
    hashmap = {feature:feature_hash for feature_hash,feature in inv_hashmap.items()}
    return hashmap,inv_hashmap

hashmap, inv_hashmap = hash_mapper(merged_data)

merged_data = merged_data.replace(hashmap)

train_data_dict = {
        agency + ' ' + sku:merged_data[(merged_data['SKU'] == hashmap[sku])&(merged_data['Agency'] == hashmap[agency])]
        for agency,sku in combs
        }

X_train, y_train, X_val, y_val = TSU.chunk_and_concatenate_dict(train_data_dict,4,1,[i for i in train_data_dict[list(train_data_dict.keys())[0]].columns if i!='Volume'],['Local_Relative_Temp','Volume'],n_validation_intervals = 5, scaler = False)

X_train = np.concatenate([X_train[:,0,:],y_train[:,0,:-1]], axis = -1)
X_val = np.concatenate([X_val[:,0,:],y_val[:,0,:-1]], axis = -1)
y_train = y_train[:,:,-1]
y_val = y_val[:,:,-1]

from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import xgboost as xgb

#from hypopt import GridSearch

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

parameters = [{
        'min_child_weight':[1],
        'max_depth':[2,4,5],        
        'early_stopping_rounds':[10],
        'booster':['gbtree'],
        'verbosity':[1],
        'subsample': [0.75],
        #'learning_rate':[0.0001,0.001,0.01,0.1],
        'eval_set': [[X_val,y_val]],
        'gamma': [0.1],
        'eval_metric': ['rmse'],
        'verbose' :[True],
        'silent' : [False],
        'min_child_weight': [1],
        'n_estimators': [100],
        'colsample_bytree': [0.1,0.2,0.3],
        'reg_lambda': [1],
        'reg_alpha': [0]
        }]
        


# Applying Grid Search to find the best model and the best parameters
#cross_v = GridSearchCV(xg_reg, parameters,scoring = 'neg_mean_squared_error',cv = 2)
#cross_v.fit(X_train,y_train)
#best_parameters = cross_v.best_params_

'''from functools import partial
from hypopt import GridSearch
grid_search = GridSearch(model = xg_reg, param_grid = parameters, cv_folds = 10)
grid_search = grid_search.fit(X_train, y_train, seller_val_set[0], seller_val_set[1],scoring = 'neg_mean_squared_error')
best_parameters = grid_search.get_params()
'''
    
xg_reg = xgb.XGBRegressor(early_stopping_rounds = 100)
best_parameters = {}
best_parameters['sublsample'] = 0.8
best_parameters['min_child_weight'] = 1
best_parameters['n_estimators'] = 1000
best_parameters['max_depth'] = 6
best_parameters['gamma'] = 0.01
best_parameters['colsample_bytree'] = 0.8
best_parameters['lambda'] = 1
best_parameters['learning_rate'] =0.1
best_parameters['objective'] = 'reg:squarederror' 
best_parameters['eval_metric'] = 'rmse'
best_parameters['eval_set'] = eval_set = [(X_train,y_train),(X_val,y_val)]

xg_reg = xgb.XGBRegressor(**best_parameters)
print(xg_reg.get_params)
#MultiOutputRegressor(xg_reg).fit(X_train,y_train,eval_set = eval_set,sample_weight = y_train,early_stopping_rounds = 10000
multi_out_model = MultiOutputRegressor(xg_reg) 
multi_out_model.fit(X_train,y_train)

plt.clf()
plt.plot(multi_out_model.predict(X_val)[:,2])
plt.plot(y_val[:,2])


preds = multi_out_model.predict(X_val)
pred_period = range(3)
rmse = np.sqrt(mean_squared_error(np.nan_to_num(y_val[:,pred_period]), preds[:,pred_period]))
print("RMSE: %f" % (rmse))

import pylab 
pylab.clf()
pylab.bar(range(len(xg_reg.feature_importances_)), xg_reg.feature_importances_)
pylab.xticks(range(len(xg_reg.feature_importances_)),[i for i in train_data_dict[list(train_data_dict.keys())[0]].columns if i!='Volume'])
pylab.show()

plt.plot(y_val)
plt.plot(preds)