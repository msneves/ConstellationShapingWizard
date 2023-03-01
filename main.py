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
from fcns.utils import sample_dist
from fcns.resetters import reset_optimizers
from fcns.plots import plot
from fcns.normalizations import norm_const
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
        
        # Normalizing APC or PPC
        const_points_n = norm_const(st.session_state.const_points,probs)

        # Mapping
        syms = tf.matmul(one_hot_syms,const_points_n)
        
        # AWGN channel
        syms_noisy = syms + tf.random.normal(shape=syms.shape,stddev=np.sqrt(st.session_state.var_1d_noise))

        # Demapper
        rx_syms = demapper(syms_noisy,probs)
        
        # Compute performance metrics 
        H, MI, GMI, MSE, CE = metrics_fcn(probs, sampled_syms_idx, one_hot_syms, rx_syms, const_points_n, syms_noisy)
        
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

    return loss_value, H, MI, GMI    


#############################################################
## Main loop ################################################
#############################################################

# While learning and the learning epochs have not been reached
while(st.session_state.learning and st.session_state.current_epoch<st.session_state.num_epochs):
    
    
    # If the number of batches  has been reached, reset optimizers
    if st.session_state.current_batch > st.session_state.num_batches:
        st.session_state.current_epoch += 1
        st.session_state.current_batch = 0
        reset_optimizers()
        
    # Else, perform batch training step 
    else:
        # Update progress bar
        st.session_state.progress.progress((st.session_state.current_batch+st.session_state.num_batches*st.session_state.current_epoch)/(st.session_state.num_batches*st.session_state.num_epochs))
        
        # Iterate over system
        st.session_state.current_loss,st.session_state.H,st.session_state.MI,st.session_state.GMI = train_step()
        st.session_state.current_batch += 1
    
    # Plot dynamic results
    if not st.session_state.current_batch%st.session_state.batches_per_plot:
        plot()
    
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
    plot()

    

    
    
    






