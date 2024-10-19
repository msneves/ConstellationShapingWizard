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
from fcns.resetters import reset_probs,reset_const,reset_demapper,reset_epochs,reset_sf,reset_ps,reset_mf

def learn_button_fcn():
    st.session_state.learning = True
    st.session_state.paused = False
    reset_epochs()

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
    reset_sf()
    reset_ps()
    reset_mf()

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
    
    initial_learning_rate = float(st.session_state.lr_probs)
    decay_steps = int(20)
    decay_rate = float(0.95)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True  
    )
    st.session_state.optimizer_log_probs = keras.optimizers.Adam(learning_rate=lr_schedule)
    reset_epochs()

def choice_const_fcn():
    if st.session_state.choice_const == 'Don\'t learn':
        st.session_state.lr_const = 0
    else:
        st.session_state.lr_const = 0.1
        reset_const()
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=float(st.session_state.lr_const),
        decay_steps=int(20),
        decay_rate=float(0.95),
        staircase=True
    )
    st.session_state.optimizer_const_points = keras.optimizers.Adam(learning_rate=lr_schedule)
    reset_epochs()

def choice_demapper_fcn():
    if st.session_state.choice_demapper == 'Min. Dist':
        st.session_state.lr_dmappr = 0
    else:
        st.session_state.lr_dmappr = 0.1
    reset_demapper()
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=float(st.session_state.lr_dmappr),
        decay_steps=int(20),
        decay_rate=float(0.95),
        staircase=True
    )
    st.session_state.optimizer_demapper = keras.optimizers.Adam(learning_rate=lr_schedule)
    reset_epochs()

def choice_ps_fcn():
    if not 'Learn' in st.session_state.choice_ps:
        st.session_state.lr_ps = 0
    else:
        st.session_state.lr_ps = 0.1
    reset_ps()
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=float(st.session_state.lr_ps),
        decay_steps=int(20),
        decay_rate=float(0.95),
        staircase=True
    )
    st.session_state.optimizer_ps = keras.optimizers.Adam(learning_rate=lr_schedule)
    reset_epochs()

def choice_mf_fcn():
    if not 'Learn' in st.session_state.choice_mf:
        st.session_state.lr_mf = 0
    else:
        st.session_state.lr_mf = 0.1
    reset_mf()
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=float(st.session_state.lr_mf),
        decay_steps=int(20),
        decay_rate=float(0.95),
        staircase=True
    )
    st.session_state.optimizer_mf = keras.optimizers.Adam(learning_rate=lr_schedule)
    reset_epochs()
        
def snr_update_fcn():
    st.session_state.var_1d_noise = 10**(-st.session_state.SNR_dB/10)/2
    reset_epochs()
    
def ntaps_update_fcn():
    st.session_state.ntaps_ps = st.session_state.ntaps
    st.session_state.ntaps_mf = st.session_state.ntaps
    reset_ps()
    reset_mf()
    reset_epochs()
    
def bw_update_fcn():
    abw = st.session_state.abw
    dbw = st.session_state.dbw
    
    if abw == 1:
        abw = .999
    
    # change to 2nd order gaussian filter
    ff = np.linspace(-.5,.5,200)
    freq_taps = np.fft.fftshift(np.exp(-np.log(np.sqrt(2))*(2/abw*ff)**(2*2)))
    freq_taps[round(dbw/2*(len(freq_taps) - 1)):round((1-dbw/2)*(len(freq_taps) - 1))] = 0.
    
    filter_taps = np.roll(np.fft.fftshift(np.fft.ifft(freq_taps)),-1)
    
    tmp_taps = np.zeros((len(filter_taps),2,2))
    tmp_taps[:,0,0] = filter_taps.real
    tmp_taps[:,1,1] = filter_taps.real
    st.session_state.taps_bw = tf.Variable(tmp_taps,dtype=tf.float32,trainable=True)
    reset_epochs()
    
    
def md_update_fcn():
    st.session_state.md_ax.clear()
    st.session_state.md_ax.axis('off')
    st.session_state.md_ax.plot(np.linspace(0,1,20),np.linspace(0,1,20),c=(.29,.29,.29), linestyle='dashed')
    st.session_state.md_ax.plot(np.linspace(0,1,20),np.sin(np.linspace(0,1,20)*np.pi/2)/np.pi*2,c=(.95,.95,.95))
    st.session_state.md_ax.vlines(x=st.session_state.md, ymin=0, ymax=np.sin(st.session_state.md*np.pi/2)/np.pi*2, linestyles='dashed',color=(.29,.29,.29))
    st.session_state.md_ax.plot([0,st.session_state.md],[np.sin(st.session_state.md*np.pi/2)/np.pi*2]*2,c=(.29,.29,.29), linestyle='dashed')
    st.session_state.md_ax.scatter(st.session_state.md,np.sin(st.session_state.md*np.pi/2)/np.pi*2,marker='o',s=100,c=[(1,0.29,0.29,1)])
    st.session_state.md_ax.text(0,np.sin(st.session_state.md*np.pi/2)/np.pi*2,f'Vppo = {np.sin(st.session_state.md*np.pi/2)/np.pi*2/st.session_state.md:.1f} Vppi',color=(.95,.95,.95))
    st.session_state.md_ax.set_xlim((0,1))
    st.session_state.md_ax.set_ylim((0,1))
    st.session_state.md_stplot.pyplot(st.session_state.md_fig)
    reset_epochs()
    
def qb_update_fcn():
    reset_const()
    reset_epochs()