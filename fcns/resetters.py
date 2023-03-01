# -*- coding: utf-8 -*-
"""
@author: M. S. Neves (msneves@ua.pt)
Instituto de Telecomunicacoes
Universidade de Aveiro
Portugal
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from fcns.qam import qam

def reset_probs():
    if st.session_state.choice_probs == 'Learn from uniform' or st.session_state.choice_probs == 'Don\'t learn':
        reset_probs = tf.ones((st.session_state.M,))/st.session_state.M
    else:
        reset_probs = np.random.randn(st.session_state.M)
    st.session_state.log_probs = tf.Variable(reset_probs, dtype=tf.float32, trainable = True, name="log_probs")
    
def reset_const():
    if st.session_state.choice_const == 'Learn from square QAM' or st.session_state.choice_const == 'Don\'t learn':
        reset_const = qam(st.session_state.M)
    else:
        reset_const = np.random.randn(st.session_state.M,2)
    st.session_state.const_points = tf.Variable(reset_const, dtype=tf.float32, trainable = True, name="const_points")
    
def reset_demapper():
    inputs = keras.Input(shape=(2,), name="noisy_syms")
    outputs = layers.Dense(st.session_state.M, activation="relu", name="rx_dense_1")(inputs)
    if st.session_state.choice_demapper=='NN Symwise':
        outputs = layers.Dense(st.session_state.M, activation="softmax", name="rx_predictions")(outputs)
    else:
        outputs = layers.Dense(np.log2(st.session_state.M), activation="sigmoid", name="rx_predictions")(outputs)
    
    st.session_state.demapper = keras.Model(inputs=inputs, outputs=outputs)
    st.session_state.optimizer_demapper = keras.optimizers.Adam(learning_rate=st.session_state.lr_dmappr)
    

def reset_optimizers():
    st.session_state.optimizer_log_probs = keras.optimizers.Adam(learning_rate=st.session_state.lr_probs)
    st.session_state.optimizer_const_points = keras.optimizers.Adam(learning_rate=st.session_state.lr_const)
    st.session_state.optimizer_demapper = keras.optimizers.Adam(learning_rate=st.session_state.lr_dmappr)
    
    
def reset_epochs():
    st.session_state.current_epoch = 0