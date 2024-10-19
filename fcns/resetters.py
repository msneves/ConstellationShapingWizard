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
from commpy.filters import rrcosfilter

def reset_probs():
    if st.session_state.choice_probs == 'Learn from uniform' or st.session_state.choice_probs == 'Don\'t learn':
        reset_probs = tf.ones((st.session_state.M,))/st.session_state.M
    else:
        reset_probs = np.random.randn(st.session_state.M)
    st.session_state.log_probs = tf.Variable(reset_probs, dtype=tf.float32, trainable = True, name="log_probs")
    reset_epochs()
    
def reset_const():
    if st.session_state.choice_const == 'Learn from square QAM' or st.session_state.choice_const == 'Don\'t learn':
        reset_const = qam(st.session_state.M)
    else:
        reset_const = np.random.randn(st.session_state.M,2)
    st.session_state.const_points = tf.Variable(reset_const, dtype=tf.float32, trainable = True, name="const_points")
    reset_epochs()
    
def reset_demapper():
    inputs = keras.Input(shape=(2,), name="noisy_syms")
    outputs = layers.Dense(st.session_state.M, activation="relu", name="rx_dense_1")(inputs)
    if st.session_state.choice_demapper=='NN Symwise':
        outputs = layers.Dense(st.session_state.M, activation="softmax", name="rx_predictions")(outputs)
    else:
        outputs = layers.Dense(np.log2(st.session_state.M), activation="sigmoid", name="rx_predictions")(outputs)
    
    st.session_state.demapper = keras.Model(inputs=inputs, outputs=outputs)
    reset_epochs()
    

def reset_optimizers():
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=st.session_state.lr_probs,
                                                              decay_steps=20,
                                                              decay_rate=0.95)
    st.session_state.optimizer_log_probs = keras.optimizers.Adam(learning_rate=lr_schedule)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=st.session_state.lr_const,
                                                              decay_steps=20,
                                                              decay_rate=0.95)
    st.session_state.optimizer_const_points = keras.optimizers.Adam(learning_rate=lr_schedule)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=st.session_state.lr_dmappr,
                                                              decay_steps=20,
                                                              decay_rate=0.95)
    st.session_state.optimizer_demapper = keras.optimizers.Adam(learning_rate=lr_schedule)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=st.session_state.lr_ps,
                                                              decay_steps=20,
                                                              decay_rate=0.95)
    st.session_state.optimizer_ps = keras.optimizers.Adam(learning_rate=lr_schedule)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=st.session_state.lr_mf,
                                                              decay_steps=20,
                                                              decay_rate=0.95)
    st.session_state.optimizer_mf = keras.optimizers.Adam(learning_rate=lr_schedule)
    
def reset_epochs():
    st.session_state.current_epoch = 0
    st.session_state.current_bach = 0
    reset_sf()

def reset_ps():
    st.session_state.ntaps_ps = st.session_state.ntaps
    if 'RRC' in st.session_state.choice_ps:
        tmp_taps = np.zeros((st.session_state.ntaps_ps,2,2))
        rrcos_taps = rrcosfilter(st.session_state.ntaps_ps+1, 0.1, 1, 2)[1][1:]
        tmp_taps[:,0,0] = rrcos_taps
        tmp_taps[:,1,1] = rrcos_taps
    elif 'square' in st.session_state.choice_ps:
        tmp_taps = np.zeros((st.session_state.ntaps_ps,2,2))
        tmp_taps[:,0,0] = [0]*(st.session_state.ntaps_ps//2) + [1,1][:st.session_state.ntaps_ps] + [0]*(st.session_state.ntaps_ps//2 - 1)
        tmp_taps[:,1,1] = [0]*(st.session_state.ntaps_ps//2) + [1,1][:st.session_state.ntaps_ps] + [0]*(st.session_state.ntaps_ps//2 - 1)
    else:
        tmp_taps = np.random.randn(st.session_state.ntaps_ps,2,2)
        
    st.session_state.taps_ps = tf.Variable(tmp_taps,dtype=tf.float32,trainable=True)
    reset_epochs()
    
def reset_mf():
    st.session_state.ntaps_mf = st.session_state.ntaps
    if 'RRC' in st.session_state.choice_mf:
        tmp_taps = np.zeros((st.session_state.ntaps_mf,2,2))
        rrcos_taps = rrcosfilter(st.session_state.ntaps_mf+1, 0.1, 1, 2)[1][1:]
        tmp_taps[:,0,0] = rrcos_taps
        tmp_taps[:,1,1] = rrcos_taps
    elif 'square' in st.session_state.choice_mf:
        tmp_taps = np.zeros((st.session_state.ntaps_mf,2,2))
        tmp_taps[:,0,0] = [0]*(st.session_state.ntaps_mf//2 - 1) + [1,1][:st.session_state.ntaps_mf] + [0]*(st.session_state.ntaps_mf//2)
        tmp_taps[:,1,1] = [0]*(st.session_state.ntaps_mf//2 - 1) + [1,1][:st.session_state.ntaps_mf] + [0]*(st.session_state.ntaps_mf//2)
    else:
        tmp_taps = np.random.randn(st.session_state.ntaps_mf,2,2)
        
    st.session_state.taps_mf = tf.Variable(tmp_taps,dtype=tf.float32,trainable=True)
    reset_epochs()
  
    
def reset_sf():
    st.session_state.sf = tf.Variable(1., dtype=tf.float32, trainable = False, name="sf")