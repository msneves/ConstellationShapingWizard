# -*- coding: utf-8 -*-
"""
@author: M. S. Neves (msneves@ua.pt)
Instituto de Telecomunicacoes
Universidade de Aveiro
Portugal
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib.colors import ListedColormap
from fcns.resetters import reset_probs,reset_const,reset_demapper,reset_epochs

def learn_button_fcn():
    st.session_state.learning = True
    st.session_state.paused = False

def stop_button_fcn():
    st.session_state.learning = False
    st.session_state.paused = True

def reset_button_fcn():
    st.session_state.learning = False
    st.session_state.paused = True
    reset_probs()
    reset_const()
    reset_demapper()
    choice_probs_fcn()
    choice_const_fcn()
    choice_demapper_fcn()

def choice_M_fcn():
    reset_probs()
    reset_const()
    reset_demapper()
    st.session_state.cmap = ListedColormap(np.random.rand(st.session_state.M,3)/2+.5)
    st.session_state.gray_labels = tf.constant([[int(x) for x in f'{i:0{int(np.log2(st.session_state.M))}b}'] for i in range(st.session_state.M)], name="gray_labels",dtype=tf.float32)
    reset_epochs()

def choice_probs_fcn():
    if st.session_state.choice_probs == 'Don\'t learn':
        st.session_state.lr_probs = 0
    else:
        st.session_state.lr_probs = .1
        reset_probs()
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=st.session_state.lr_probs,
                                                              decay_steps=20,
                                                              decay_rate=0.95)
    st.session_state.optimizer_log_probs = keras.optimizers.Adam(learning_rate=lr_schedule)
    reset_epochs()

def choice_const_fcn():
    if st.session_state.choice_const == 'Don\'t learn':
        st.session_state.lr_const = 0
    else:
        st.session_state.lr_const = .1
        reset_const()
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=st.session_state.lr_const,
                                                              decay_steps=20,
                                                              decay_rate=0.95)
    st.session_state.optimizer_const_points = keras.optimizers.Adam(learning_rate=lr_schedule)
    reset_epochs()

def choice_demapper_fcn():
    if st.session_state.choice_demapper == 'Min. Dist':
        st.session_state.lr_dmappr = 0
    else:
        st.session_state.lr_dmappr = .1
    reset_demapper()
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=st.session_state.lr_dmappr,
                                                              decay_steps=20,
                                                              decay_rate=0.95)
    st.session_state.optimizer_demapper = keras.optimizers.Adam(learning_rate=lr_schedule)
    reset_epochs()
        
def snr_update_fcn():
    st.session_state.var_1d_noise = 10**(-st.session_state.SNR_dB/10)/2
    reset_epochs()