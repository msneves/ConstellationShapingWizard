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
from fcns.webapp import page_setup
from fcns.utils import sample_dist,gen_rpn,quantize,apply_md
from fcns.plots import plot_const,plot_filters
from fcns.normalizations import norm,norm_const,sig_pow,const_pow
from fcns.filters import pulse_shape,matched_filter,channel_bw_limitation
from fcns.demapper import demapper
from fcns.losses import metrics_fcn

#############################################################
## Page Setup ###############################################
#############################################################
st.session_state = page_setup()

#############################################################
## TX/RX System #############################################
#############################################################

# Function that simulates the transmission system and estimates resulting metrics 
@tf.function
def train_step():
    
    # A gradient tape allows to record the gradients of the operations performed seamlessly
    with tf.GradientTape(persistent=True) as tape:
        
        # From unconstrained log probs to probs
        probs = tf.nn.softmax(st.session_state.log_probs)

        # Sampling from probs
        sampled_syms_idx = sample_dist(st.session_state.batch_size,probs)

        # Turning the indexes into one-hot vectors
        one_hot_syms = tf.one_hot(sampled_syms_idx, st.session_state.M)       
        
        # Mapping
        const_points_n = norm_const(st.session_state.const_points,probs)
        syms = tf.matmul(one_hot_syms,const_points_n)
        
        # Upsampling + Pulse Shaping
        syms_ps = pulse_shape(syms)
            
        # Channel starts here
        
        # # Quantizing
        syms_channel = quantize(syms_ps, st.session_state.qb)
        
        # Normalization and Modulation
        if st.session_state.norm_mode == 'APC':
            # If APC, first apply nonlinearity then normalize 
            syms_channel = apply_md(syms_channel,st.session_state.md)
            syms_channel = syms_channel*tf.sqrt(const_pow(const_points_n,probs)/const_pow(apply_md(const_points_n,st.session_state.md),probs))
        else:
            # If PPC, just apply nonlinearity 
            syms_channel = apply_md(syms_channel,st.session_state.md)
        
        # Transceiver+Channel Filtering
        syms_channel = channel_bw_limitation(syms_channel)  
        
        # Residual LPN noise
        syms_channel = syms_channel + gen_rpn(syms_channel,st.session_state.var_phase_noise)
        
        # AWGN channel
        syms_channel = syms_channel + tf.random.normal(shape=syms_channel.shape,stddev=np.sqrt(2*st.session_state.var_1d_noise))
        
        # Quantizing
        syms_channel = quantize(syms_channel, st.session_state.qb)
        
        # Matched Filter + Downsampling
        syms_channel = matched_filter(syms_channel)
        
        # Normalize signal 
        syms_channel = norm(syms_channel)
        
        # Demapper
        n_var = sig_pow(syms-syms_channel)
        rx_syms = demapper(syms_channel,probs,n_var)
        
        # Compute performance metrics
        H, MI, GMI, MSE, CE = metrics_fcn(probs, sampled_syms_idx, one_hot_syms, n_var, rx_syms, const_points_n, syms_channel)
        
        st.session_state.H = H
        st.session_state.MI = MI
        st.session_state.GMI = GMI
        st.session_state.MSE = MSE
        st.session_state.CE = CE
        
        # Attribute loss function to minimize
        if st.session_state.loss_fcn=='MSE':
            loss_value = MSE
        elif st.session_state.loss_fcn=='CE':
            loss_value = CE
        elif st.session_state.loss_fcn=='GMI':
            loss_value = - GMI
        else:
            loss_value = - MI
   
    # Update probabilities 
    if st.session_state.lr_probs>0:
        grads_log_probs = tape.gradient(loss_value, st.session_state.log_probs)
        if grads_log_probs is not None:
            st.session_state.optimizer_log_probs.apply_gradients([(grads_log_probs, st.session_state.log_probs)])
    
    # Update constellation points
    if st.session_state.lr_const>0:
        grads_const_points = tape.gradient(loss_value, st.session_state.const_points)
        st.session_state.optimizer_const_points.apply_gradients([(grads_const_points, st.session_state.const_points)])
    
    # Update Demapper coefficients
    if st.session_state.lr_dmappr>0:
        grads_demapper = tape.gradient(loss_value, st.session_state.demapper.trainable_weights)
        st.session_state.optimizer_demapper.apply_gradients(zip(grads_demapper, st.session_state.demapper.trainable_weights))

    # Update Pulse Shaper
    if st.session_state.lr_ps>0:
        grads_ps = tape.gradient(loss_value, st.session_state.taps_ps)
        st.session_state.optimizer_ps.apply_gradients([(grads_ps, st.session_state.taps_ps)])
    
    # Update Matched Filter
    if st.session_state.lr_mf>0:
        grads_mf = tape.gradient(loss_value, st.session_state.taps_mf)
        st.session_state.optimizer_mf.apply_gradients([(grads_mf, st.session_state.taps_mf)])
        
    print('taps_mf',st.session_state.taps_mf)

    return loss_value, H, MI, GMI, syms_channel,n_var


#############################################################
## Main loop ################################################
#############################################################

# While learning and the learning epochs have not been reached
while(st.session_state.learning and st.session_state.current_epoch<st.session_state.num_epochs):
    
    
    # If the number of batches  has been reached, reset optimizers
    if st.session_state.current_batch > st.session_state.num_batches:
        st.session_state.current_epoch += 1
        st.session_state.current_batch = 0
        
    # Else, perform batch training step 
    else:
        # Update progress bar
        st.session_state.progress.progress((st.session_state.current_batch+st.session_state.num_batches*st.session_state.current_epoch)/(st.session_state.num_batches*st.session_state.num_epochs))
        
        # Iterate over system
        st.session_state.current_loss,st.session_state.H,st.session_state.MI,st.session_state.GMI,sig,n_var = train_step()
        st.session_state.current_batch += 1
    
    # Plot dynamic results
    if not st.session_state.current_batch%st.session_state.batches_per_plot:
        plot_const(sig,n_var)
        plot_filters()
        st.session_state.sig = sig
        st.session_state.n_var = n_var
    
    # Log status:
    print(f'SESS{st.session_state.session_id}:EP{st.session_state.current_epoch}:BA{st.session_state.current_batch}:LO{st.session_state.current_loss}')


#############################################################
## Idle State ###############################################
#############################################################

# If num_epochs has been reached, pause learning
if st.session_state.learning and st.session_state.current_epoch == st.session_state.num_epochs:
    st.session_state.learning = False
    st.session_state.paused = True
    st.experimental_rerun()

# Plot current results when paused
if st.session_state.paused:
    plot_const(st.session_state.sig,st.session_state.n_var)
    plot_filters()
    st.session_state.progress.progress(0)