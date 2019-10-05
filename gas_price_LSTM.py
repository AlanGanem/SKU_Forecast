# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 21:12:08 2019

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

latent_dim = 2
look_back_period = 16
pred_period = 8
dropout_rate = 0
batch_size = 128
#read data and convert to np array
data = pd.read_table(r'C:\Users\ganem\Desktop\Arquivos pessoais\Projetos\Kaggle\gas_prices_brazil\2004-2019.tsv',delim_whitespace=False,header=0)
data['DATA INICIAL'] = pd.to_datetime(data['DATA INICIAL'])
data['DATA FINAL'] = pd.to_datetime(data['DATA FINAL'])
rng=  pd.date_range(data['DATA FINAL'].min(), data['DATA FINAL'].max()+pd.Timedelta('1 days'), freq='W')

data.set_index('DATA FINAL',inplace = True)
data.replace('-',np.nan, inplace = True)
estados_dummies = ['ESTADO_'+estado for estado in data['ESTADO'].unique()]
produtos_dummies = ['PRODUTO_'+produto for produto in data['PRODUTO'].unique()]
mes_dummies = ['MÊS_'+ str(mes) for mes in data['MÊS'].unique()]
dummie_columns = estados_dummies+produtos_dummies
train_data = pd.get_dummies(data,columns = ['ESTADO','MÊS','PRODUTO'])

number_columns = []
for column in data.columns:
    try:
        data[column] = data[column].astype(float)     
        number_columns.append(column)
        print(column)
    except:    
        pass    

features = ['PREÇO MÉDIO REVENDA', 'DESVIO PADRÃO REVENDA', 'PREÇO MÍNIMO REVENDA', 'PREÇO MÁXIMO REVENDA', 'MARGEM MÉDIA REVENDA', 'COEF DE VARIAÇÃO REVENDA', 'PREÇO MÉDIO DISTRIBUIÇÃO', 'DESVIO PADRÃO DISTRIBUIÇÃO', 'PREÇO MÍNIMO DISTRIBUIÇÃO', 'PREÇO MÁXIMO DISTRIBUIÇÃO', 'COEF DE VARIAÇÃO DISTRIBUIÇÃO']
features_cut = len(dummie_columns)
future_features = ['MÊS']
features = dummie_columns + features

dependent_variable = ['PREÇO MÉDIO REVENDA']
try:
    features.pop(features.index(dependent_variable[0]))
except IndexError:
    pass
features = features+dependent_variable
train_data = train_data[features]
train_data = train_data.fillna(method = 'ffill', limit = 300)

train_data[features]

combs = list(itertools.product(estados_dummies,produtos_dummies))

train_data_dict = {estado+' '+produto:train_data[(train_data[produto] == 1)&(train_data[estado] == 1)][features].append(pd.DataFrame(index = rng)).astype('float').resample('W').median().fillna(method = 'ffill', limit= 3).fillna(value = 0)[features] for estado,produto in combs}
        
assert all([i in train_data_dict[list(train_data_dict.keys())[0]].columns for i in features])


covars = False
if covars:
    i=0
    for key in train_data_dict.keys():
        data_arr = train_data_dict[key].values
        if i == 0:
            #data_arr = minmax_scaler.fit_transform(data_arr)
            X = TSU.chunk_data_by_date(data_arr,pred_period,look_back_period)
            X_train, y_train, X_val, y_val = X
            X_cov_train,X_cov_val = X_train[:,:pred_period,features_cut:],X_val[:,:pred_period,features_cut:]
            X_train, y_train, X_val, y_val = X_train[:,:,:features_cut], y_train, X_val[:,:,:features_cut], y_val
        else:
            X_ = TSU.chunk_data_by_date(data_arr,pred_period,look_back_period)
            X_train_, y_train_, X_val_, y_val_ = X_
            X_cov_train_ = X_train_[:,:pred_period,features_cut:]
            X_cov_val_ = X_val_[:,:pred_period,features_cut:]
            X_train_ = X_train_[:,:,:features_cut]
            X_val_ = X_val_[:,:,:features_cut]
            
            X = np.concatenate((X,X_))
            X_train = np.concatenate((X_train,X_train_))
            X_cov_train = np.concatenate((X_cov_train,X_cov_train_))
            y_train = np.concatenate((y_train,y_train_))
            X_val = np.concatenate((X_val,X_val_ ))
            X_cov_val = np.concatenate((X_cov_val,X_cov_val_ ))
            y_val = np.concatenate((y_val,y_val_))        
        i+=1
    X_train_no_teacher_forcing, X_val_no_teacher_forcing = X_cov_train, X_cov_val
else:
    i=0
    for key in train_data_dict.keys():
        data_arr = train_data_dict[key].values
        if i == 0:
            #data_arr = minmax_scaler.fit_transform(data_arr)
            X = TSU.chunk_data_by_date(data_arr,pred_period,look_back_period)
            X_train, y_train, X_val, y_val = X
            
            X_train_teacher_forcing,X_val_teacher_forcing,X_train_no_teacher_forcing,X_val_no_teacher_forcing = TSU.teacher_forcing_generator(y_train,y_val)
        else:
            X_ = TSU.chunk_data_by_date(data_arr,pred_period,look_back_period)
            X_train_, y_train_, X_val_, y_val_ = X_
            X_train_teacher_forcing_,X_val_teacher_forcing_,X_train_no_teacher_forcing_,X_val_no_teacher_forcing_ = TSU.teacher_forcing_generator(y_train_,y_val_)            
            
            X = np.concatenate((X,X_))
            X_train = np.concatenate((X_train,X_train_))
            X_train_teacher_forcing = np.concatenate((X_train_teacher_forcing,X_train_teacher_forcing_))
            X_val_teacher_forcing = np.concatenate((X_val_teacher_forcing,X_val_teacher_forcing_))
            X_train_no_teacher_forcing = np.concatenate((X_train_no_teacher_forcing,X_train_no_teacher_forcing_))
            X_val_no_teacher_forcing = np.concatenate((X_val_no_teacher_forcing,X_val_no_teacher_forcing_))
            
            y_train = np.concatenate((y_train,y_train_))
            X_val = np.concatenate((X_val,X_val_ ))
            y_val = np.concatenate((y_val,y_val_))        
        i+=1

assert not np.any(np.isnan(X_train))
assert not np.any(np.isnan(X_val))
assert not np.any(np.isnan(X_train_no_teacher_forcing))
assert not np.any(np.isnan(X_train_teacher_forcing))



# defininf layers
encoder_input = Input(shape = (X_train.shape[1],X_train.shape[2]),name = 'encoder_input')
decoder_input = Input(shape = (X_train_no_teacher_forcing.shape[1],X_train_no_teacher_forcing.shape[2]),name = 'decoder_input')

encoderLSTM = LSTM(units = latent_dim,return_state = True,return_sequences = True,name = 'enc_LSTM',dropout = dropout_rate)
attention1 = AttentionLayer()
decoderLSTM = LSTM(units = latent_dim,return_state = True,return_sequences = True,name = 'dec_LSTM',dropout = dropout_rate)
dense_output = TimeDistributed(Dense(1),name = 'time_distirbuted_dense_output')
gaussian_layer = GaussianLayer(1)

## building model

encoder_out, encoder_states = encoderLSTM(encoder_input)[0], encoderLSTM(encoder_input)[1:]

decoder_out, decoder_states = decoderLSTM(decoder_input,initial_state = encoder_states)[0],decoderLSTM(decoder_input,initial_state = encoder_states)[1:]

output = dense_output(decoder_out)

# explicitly define tensor shapes as TensorShape([Dimension(128), Dimension(12), Dimension(10)]) and TensorShape([Dimension(128), Dimension(8), Dimension(10)])

attn_out, attn_states = attention1([encoder_out,decoder_out])

decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

dense = Dense(1, activation='linear', name='output_layer')
dense_time = TimeDistributed(dense, name='time_distributed_layer')
decoder_pred = dense_time(decoder_concat_input)
mu, sigma = gaussian_layer(decoder_pred)

model = Model(inputs=[encoder_input, decoder_input], outputs=mu)
model.summary()

model.compile(optimizer = 'adam',loss = gaussian_likelihood(sigma))
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30,restore_best_weights = True)
#train_history = model.fit([X_train,X_train_no_teacher_forcing], y_train, batch_size = 128, epochs = 5000,validation_data = [[X_val,X_val_no_teacher_forcing], y_val],callbacks = [es])
train_history = model.fit([X_train,X_train_no_teacher_forcing], y_train, batch_size = 512, epochs = 5000, validation_data =[[X_val,X_val_no_teacher_forcing], y_val] ,callbacks = [es])

plt.clf()
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])

pred_sigma_mu = Model(inputs = [encoder_input,decoder_input],outputs = [mu,sigma])
sigma_mu_preds = pred_sigma_mu.predict([X_val,X_val_no_teacher_forcing])
sigma_mu_preds_list = [m_s for m_s in zip(sigma_mu_preds[0].flatten(),sigma_mu_preds[1].flatten())]
plt.clf()
plt.plot(y_val[0,:])
plt.plot(sigma_mu_preds[0][0,:])
plt.plot(sigma_mu_preds[0][0,:]+3*sigma_mu_preds[1][0,:])
plt.plot(sigma_mu_preds[0][0,:]-3*sigma_mu_preds[1][0,:])





enc_pred_model = Model(encoder_input,[encoder_out]+encoder_states)

dec_input_states = [Input(shape = (latent_dim,)),Input(shape = (latent_dim,))]

dec_outputs_and_states = decoderLSTM(decoder_input, initial_state = dec_input_states)
dec_outputs = dec_outputs_and_states[0]
dec_states = dec_outputs_and_states[1:]

attn_outputs_p, attn_states_p = attention1([encoder_out,decoder_out])


dec_pred_model = Model([decoder_input] + dec_input_states,[dec_outputs]+ dec_states)
attn_inputs = [Input(shape = (look_back_period,latent_dim)),Input(shape = (pred_period,latent_dim))]
attn_layer_out= attention1(attn_inputs)
attn_model = Model(attn_inputs,attn_layer_out)

final_layer_in = [Input(shape = (pred_period,latent_dim)),Input(shape = (pred_period,latent_dim))]
final_layer_out = Concatenate(axis=-1)(final_layer_in)
final_layer_out = dense_time(final_layer_out)
final_layer_model = Model(final_layer_in,final_layer_out)

y_train,y_val = y_train/scale_factor,y_val/scale_factor
preds = TSU.enc_dec_predict(X_val,enc_pred_model,dec_pred_model,pred_period,latent_dim)
train_preds = TSU.enc_dec_predict(X_train,enc_pred_model,dec_pred_model,pred_period,latent_dim)

train_preds_avg = TSU.average_anti_diag(train_preds)
preds_avg = TSU.average_anti_diag(preds)

plt.plot((range(8)),model.predict([X_val[1:2], X_val_no_teacher_forcing[-1:]])[0])
plt.plot((range(8)),model.predict([X_val[1:2], X_val_no_teacher_forcing[-1:]])[0])
plt.plot((range(8)),y_val[2])


for i in X_train.shape[0]:
    X_train[1,:,:X_train.shape[2]-1].shape


