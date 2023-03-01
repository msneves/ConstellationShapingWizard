# -*- coding: utf-8 -*-
"""
@author: M. S. Neves (msneves@ua.pt)
Instituto de Telecomunicacoes
Universidade de Aveiro
Portugal
"""

import tensorflow as tf
from tensorflow import keras
import streamlit as st
from fcns.utils import gaussian2d_pdf,bit2sym_likelihoods,sym2bit_likelihoods

# Function to compute mutual information (MI) and generalized MI (GMI)
def gmi_eval(probs,x,x_onehot,y,const_points_n,syms_noisy):
    
    # Entropy
    H = - tf.reduce_sum(probs*tf.math.log(probs+tf.keras.backend.epsilon())/tf.math.log(tf.constant(2,dtype=tf.float32)))
    
    # Analytical -- Assumes Gaussian Dist 
    P_Y_X = gaussian2d_pdf(st.session_state.var_1d_noise, const_points_n, syms_noisy)
    
    ## For MI
    if 'Bitwise' in st.session_state.choice_demapper:
        # if demapper works on the bit level, obtain symbol likelihoods from bit likelihoods
        y_sym = bit2sym_likelihoods(y)
    else:
        y_sym = y
    
    sum_log2_Q = tf.reduce_sum(P_Y_X*tf.math.log(y_sym + tf.keras.backend.epsilon())/tf.math.log(tf.constant(2,dtype=tf.float32)),axis=0)
    
    X_mi = - tf.reduce_sum(probs*sum_log2_Q)
    
    ## For GMI
    if not 'Bitwise' in st.session_state.choice_demapper:
        # if demapper works on the symbol level, obtain bit likelihoods from symbol likelihoods
        y = sym2bit_likelihoods(y)
    
    P_bk_Y = tf.stack([1-y,y],axis=-1)
    
    P_Y_bk_cj = P_Y_X*probs
    
    P_Y_bk =  tf.reduce_sum(tf.expand_dims(tf.stack([1-st.session_state.gray_labels,st.session_state.gray_labels],axis=-1),axis=0)*tf.expand_dims(tf.expand_dims(P_Y_bk_cj,axis=-1),axis=-1),axis=1)
    P_Y_bk /= tf.reduce_sum(P_Y_bk,axis=0) + tf.keras.backend.epsilon()
    
    sum_log2_Q = tf.reduce_sum(P_Y_bk*tf.math.log(P_bk_Y + tf.keras.backend.epsilon())/tf.math.log(tf.constant(2,dtype=tf.float32)),axis=0)
    
    P_bk = tf.reduce_sum(tf.stack([1-st.session_state.gray_labels,st.session_state.gray_labels],axis=-1)*tf.expand_dims(tf.expand_dims(probs,axis=-1),axis=-1),axis=0)
    
    X_gmi = - tf.reduce_sum(P_bk*sum_log2_Q)
    
    return H, H - X_mi, H - X_gmi

# Function to compute mean squared error (MSE)
def mse_eval(tx_y,rx_y):
    return keras.losses.MeanSquaredError()(tx_y,rx_y)

# Function to compute categorical cross-entropy (CCE)
def cce(tx_y,rx_y):
    return keras.losses.CategoricalCrossentropy()(tx_y,rx_y)

# Function to compute binary cross-entropy (BCE)
def bce(tx_y,rx_y):
    return keras.losses.BinaryCrossentropy()(tx_y,rx_y)

# Function that, given a batch, returns the metrics for it, namely H, MI, GMI, MSE and CE
def metrics_fcn(probs,x,x_onehot,y,const_points_n,syms_noisy):
    
    H,MI,GMI = gmi_eval(probs, x, x_onehot, y, const_points_n, syms_noisy)
    
    if not 'Bitwise' in st.session_state.choice_demapper:     
        MSE = mse_eval(x_onehot, y)
        CE = cce(x_onehot, y)
    else:
        # If the demapper is bitwise, we need to code the transmitted labels to bits
        x_bits = tf.gather(st.session_state.gray_labels,x,axis=0)
        
        MSE = mse_eval(x_bits, y)
        CE = bce(x_bits, y)
    
    return H, MI, GMI, MSE, CE

