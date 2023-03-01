# -*- coding: utf-8 -*-
"""
@author: M. S. Neves (msneves@ua.pt)
Instituto de Telecomunicacoes
Universidade de Aveiro
Portugal
"""

import streamlit as st
import tensorflow as tf
from fcns.utils import gaussian2d_pdf
from fcns.normalizations import norm_const

# This function implements the standard maximum a posteriori (MAP) demapper
def demapper_std(rx_sig,probs):
    ref_const = norm_const(st.session_state.const_points, probs)
    
    # Used to compute P(X|Y)
    P_Y_X = gaussian2d_pdf(st.session_state.var_1d_noise,ref_const,rx_sig)
    
    P_X_Y = P_Y_X*probs/(tf.reduce_sum(P_Y_X*probs,axis=-1,keepdims=True) + tf.keras.backend.epsilon())
    
    return P_X_Y

# This function selects the type of demapper to use, either NN ot MAP demapper
def demapper(rx_syms, probs):
    if 'NN' in st.session_state.choice_demapper:
        rx_syms = st.session_state.demapper(rx_syms, training=True)
    else:
        rx_syms = demapper_std(rx_syms, probs)
        
    return rx_syms