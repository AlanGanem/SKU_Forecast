# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 22:25:57 2019

@author: ganem
"""


enc_in = X_val[0:1]

dec_in = np.zeros((1,8,1))

enc_outs = enc_pred_model.predict(enc_in)
enc_out = enc_outs[0]
enc_states = enc_outs[1:]
dec_outs= dec_pred_model.predict([dec_in]+enc_states)
dec_out = dec_outs[0]
dec_states  = dec_outs[1:]
attn_outs= attn_model([enc_out, dec_out])
attn_out = attn_outs[0]
attn_states = attn_outs[1:]
output = final_layer_model([dec_out,attn_out])
enc_in[:,-1,-1](np.take(output,axis = -2, indices = 0)[0][0])
encoder_input = encoder_input[1:]

