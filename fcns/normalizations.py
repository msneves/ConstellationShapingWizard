# -*- coding: utf-8 -*-
"""
@author: M. S. Neves (msneves@ua.pt)
Instituto de Telecomunicacoes
Universidade de Aveiro
Portugal
"""

import tensorflow as tf
import streamlit as st

# This function normallizes the constellation with an average or peak power constraint
def norm_const(const,probs):
    if st.session_state.norm_mode == 'APC':
        return const/tf.math.sqrt(const_pow(const,probs))
    else:
        return const/(tf.math.reduce_max(tf.math.abs(const),axis=0))

# This function computes the power of the constellation, weigthed with the probabilities
def const_pow(const,probs):
    return tf.math.reduce_sum(probs*tf.math.reduce_sum(const**2,axis=-1))


