# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:49:07 2019

@author: ganem
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:54:49 2019

@author: ganem
"""

import tensorflow as tf
from attention import AttentionLayer
from tensorflow import metrics
from tensorflow.python.keras.layers import Input, GRU, Dense, Concatenate, TimeDistributed
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Dropout, Concatenate ,Dense ,LSTM,MaxPooling1D ,Conv1D, MaxPooling1D,Input, TimeDistributed, Flatten, Conv2D,Reshape,Permute, Flatten
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.utils import plot_model
from deepar.model.layers import GaussianLayer
from deepar.model.loss import gaussian_likelihood
import TimeSeriesUtils as TSU
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import inspect
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
import itertools
from tensorflow.python.keras import constraints

look_back_period = 12
pred_period = 3

latent_dim = 8
dropout_rate = 0.2
batch_size = 1024

X_train, y_train, X_val, y_val = TSU.chunk_and_concatenate_dict(train_data_dict,pred_period,look_back_period,encoder_inputs,decoder_inputs,n_validation_intervals = 2, scaler = False)
X_covars_train = y_train.take(range(y_train.shape[-1]-1),axis = -1)
X_covars_val = y_val.take(range(y_val.shape[-1]-1),axis = -1)
y_train = y_train.take([-1],axis = -1)
y_val = y_val.take([-1],axis = -1)

assert not np.any(np.isnan(X_train))
assert not np.any(np.isnan(X_val))
assert not np.any(np.isnan(X_covars_val))
assert not np.any(np.isnan(X_covars_train))
assert not np.any(np.isnan(y_val))
assert not np.any(np.isnan(y_train))



encoder_input = Input(shape = (X_train.shape[1],X_train.shape[2]),name = 'encoder_input')
decoder_input = Input(shape = (X_covars_train.shape[1],X_covars_train.shape[2]),name = 'decoder_input')

encoderLSTM = LSTM(units = latent_dim,return_state = True,return_sequences = True,name = 'enc_LSTM',dropout = dropout_rate)
decoderLSTM = LSTM(units = latent_dim,return_state = True,return_sequences = True,name = 'dec_LSTM',dropout = dropout_rate)
dense_output = TimeDistributed(Dense(1),name = 'time_distirbuted_dense_output')


## building model

encoder_out, encoder_states = encoderLSTM(encoder_input)[0], encoderLSTM(encoder_input)[1:]

decoder_out, decoder_states = decoderLSTM(decoder_input,initial_state = encoder_states)[0],decoderLSTM(decoder_input,initial_state = encoder_states)[1:]

# explicitly define tensor shapes as TensorShape([Dimension(128), Dimension(12), Dimension(10)]) and TensorShape([Dimension(128), Dimension(8), Dimension(10)])

dense = Dense(1, activation='relu', name='output_layer', kernel_constraint = 
              constraints.NonNeg())
dense_time = TimeDistributed(dense, name='time_distributed_layer')
decoder_pred = dense_time(decoder_out)
mu = decoder_pred
model = Model(inputs=[encoder_input, decoder_input], outputs=mu)
model.summary()

model.compile(optimizer = 'adam',loss = 'mse' ,sample_weight_mode = 'temporal')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30,restore_best_weights = True)

train_history = model.fit([X_train,X_covars_train], y_train, batch_size = batch_size, epochs = 5000, validation_data =[[X_val,X_covars_val], y_val] ,callbacks = [es],sample_weight = y_train[:,:,0])


plt.clf()
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])

df_transformer = data_transformer(train_data_dict,key,pred_period,look_back_period,encoder_inputs,decoder_inputs)
pred_sigma_mu = Model(inputs = [encoder_input,decoder_input],outputs = mu)
sigma_mu_preds = pred_sigma_mu.predict([df_transformer['X_train'],df_transformer['X_covars_train']])
sigma_mu_preds_list = [m_s for m_s in zip(sigma_mu_preds[0].flatten(),sigma_mu_preds[1].flatten())]
plt.clf()
i = 0
i+=1
plt.plot(range(i,len(y_train[i,:])+i),df_transformer['y_train'][i,:],color = 'b')
#plt.plot(range(i,len(y_train[i,:])+i),sigma_mu_preds[i])
plt.plot(range(i,len(y_train[i,:])+i),sigma_mu_preds[i], color = 'orange')
