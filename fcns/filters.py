# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:17:20 2023

@author: msneves
"""
import tensorflow as tf
import streamlit as st
import numpy as np
from fcns.normalizations import sig_pow


# This function performs pulse shaping on the signal
def pulse_shape(sig):  
    # Get initial sig power
    sigpow_i = sig_pow(sig)
    # Upsample the signal
    sig = _resample(sig, 2, 1)
    
    # Filter with pulse shaper taps
    filtered_sig = _filter(sig, st.session_state.taps_ps)
    
    # Restore sig power
    filtered_sig *= tf.sqrt(sigpow_i/sig_pow(filtered_sig))
    
    return filtered_sig

# This function performs matched filtering on the signal
def matched_filter(sig):
    # Get initial sig power
    sigpow_i = sig_pow(sig)
    
    # Filter with matched filter taps
    sig = _filter(sig, st.session_state.taps_mf)
    
    # Upsample the signal
    filtered_sig = _resample(sig, 1, 2)
    
    # Restore sig power
    filtered_sig *= tf.sqrt(sigpow_i/sig_pow(filtered_sig))
    
    return filtered_sig

def channel_bw_limitation(sig):                     
    # Filter with channel filter   
    return _filter(sig, st.session_state.taps_bw)
    

# This function performs time-domain filtering of a signal 
def _filter(sig,taps):
    # Add an extra dimension to the signal to match the input requirements of tf.nn.conv1d
    sig = tf.expand_dims(sig, axis=0)
    taps = tf.expand_dims(taps, axis=-1)
    tmp_taps = tf.zeros_like(taps)
    tmp_taps = tf.concat([tf.concat([taps, tmp_taps], axis=1),tf.concat([tmp_taps, taps], axis=1)], axis=2)
    
    # Perform the convolution to apply the filter
    filtered_sig = tf.nn.conv1d(sig, tmp_taps, stride=1, padding='SAME')
    
    # Remove the extra dimension added to the signal
    filtered_sig = tf.squeeze(filtered_sig, axis=0)
        
    return filtered_sig


# This function resamples the signal following the ratio m/n
# m or n must be a multiple of the other 
def _resample(sig,m,n):
    upsampled_shape = sig.shape.as_list()
    upsampled_shape[0] *= m 
    indices = np.arange(0,upsampled_shape[0],m)[:,None]
    upsampled_sig = tf.scatter_nd(indices, sig, upsampled_shape)
    resampled_sig = upsampled_sig[::n,...]
    return resampled_sig


