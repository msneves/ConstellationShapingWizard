# -*- coding: utf-8 -*-
"""
@author: M. S. Neves (msneves@ua.pt)
Instituto de Telecomunicacoes
Universidade de Aveiro
Portugal
"""

import tensorflow as tf
import streamlit as st
import numpy as np

# Convert the output of a symwise demapper into the output of a bitwise demapper
def sym2bit_likelihoods(sym_likelihoods):
    out = tf.reduce_sum(tf.expand_dims(st.session_state.gray_labels,axis=0)*tf.expand_dims(sym_likelihoods,axis=-1),axis=1)
    return tf.math.minimum(out,1)

# Convert the output of a bitwise demapper into the output of a symwise demapper
def bit2sym_likelihoods(bit_likelihoods):
    out = tf.reduce_prod(tf.reduce_sum(tf.expand_dims(tf.stack([1-bit_likelihoods,bit_likelihoods],axis=-1),axis=1)*tf.expand_dims(tf.stack([1-st.session_state.gray_labels,st.session_state.gray_labels],axis=-1),axis=0),axis=-1),axis=-1)
    return out/tf.reduce_sum(out,axis=-1,keepdims=True)

# Function that returns the P(Y|X), for all possible received symbols Y
def gaussian2d_pdf(var_noise,ref_const,rx_sig):
    P_Y_X = 1./(2.*np.pi*var_noise)*tf.math.exp(-1./(2.*var_noise)*tf.reduce_sum((tf.transpose(tf.expand_dims(ref_const,axis=0),perm=[0,2,1])-tf.expand_dims(rx_sig,axis=-1))**2,axis=-2))
    return P_Y_X/tf.reduce_sum(P_Y_X,axis=0)

# This function gets symbol indexes given a list of symbol probabilities
def sample_dist(num_syms,probs):
    cumulative_distncs = tf.cumsum(probs)
    rand_unif = tf.random.uniform(shape=(num_syms,), minval=0.0, maxval=tf.reduce_max(cumulative_distncs), dtype=tf.float32)
    greater_than = tf.expand_dims(rand_unif, axis=-1)>tf.expand_dims(cumulative_distncs, axis=0)
    return tf.reduce_sum(tf.cast(greater_than, dtype=tf.int64), axis=1)