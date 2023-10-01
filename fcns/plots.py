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
import pandas as pd
from fcns.demapper import demapper
from fcns.normalizations import sig_pow
import matplotlib.pyplot as plt
from time import sleep

def plot(sig,n_var):
    st.session_state.ax.clear()
    
    probs = tf.nn.softmax(st.session_state.log_probs).numpy()

    tc = (st.session_state.const_points/tf.math.sqrt(tf.math.reduce_sum(tf.nn.softmax(st.session_state.log_probs)*tf.math.reduce_sum(st.session_state.const_points**2,axis=-1)))).numpy()
    tl = np.array([f'{i:06b}' for i in range(st.session_state.M)],dtype=str).reshape((st.session_state.M,1))
    
    ppi = 120
    nx, ny = (ppi, ppi)
    x = np.linspace(-2, 2, nx).astype(np.float32)
    y = np.linspace(-2, 2, ny).astype(np.float32)

    xv, yv = np.meshgrid(x, y)

    coords_prompt = np.stack((xv.flatten(),yv.flatten())).T

    if st.session_state.choice_demapper == 'NN Bitwise':
        res = np.round(demapper(coords_prompt,probs,n_var).numpy()).astype(int)
        res = np.array([int("".join(str(x) for x in res[k,:]), 2) for k in range(res.shape[0])])
    else:
        res = tf.argmax(demapper(coords_prompt, probs,n_var),axis=-1).numpy()
        
    grid = np.reshape(res,(ppi,ppi))
      
    
    st.session_state.ax.pcolormesh(x,y,grid,shading='gouraud',cmap=st.session_state.cmap)

    if sig is not None:
        st.session_state.ax.scatter(sig[:,0],sig[:,1],marker='.',color='blue',s=.1)
    for i in range(st.session_state.M):
        st.session_state.ax.scatter(tc[i,0],tc[i,1],marker='o',color='red',s=2*(probs[i]*100)**2)
    if st.session_state.show_probs:
        for i in range(st.session_state.M):
            st.session_state.ax.text(tc[i,0],tc[i,1],f'{probs[i]*100:.2f}', clip_on=True)
    if st.session_state.show_bitlabels:
        for i in range(st.session_state.M):
            st.session_state.ax.text(tc[i,0],tc[i,1]-.12,f'{i:0{int(np.log2(st.session_state.M))}b}', clip_on=True,fontsize=6)
        
    st.session_state.ax.set_xlim((-2,2))
    st.session_state.ax.set_ylim((-2,2))
    st.session_state.ax.set_title(f'H = {st.session_state.H:0.2f}, ' + \
                                  f'MI = {max(0,st.session_state.MI):0.2f}, ' + \
                                  f'GMI = {max(0,st.session_state.GMI):0.2f}, ' + \
                                  f'Teor. Shan. Lim. = {np.log2(1+1/(2*st.session_state.var_1d_noise)):0.2f}, ' + \
                                  f'SNR_eff = {10*np.log10(sig_pow(sig)/n_var):0.2f} dB, ' + \
                                  f'Eff. Shan. Lim. = {np.log2(1+1/n_var):0.2f}' \
                                    , color='white')
    st.session_state.ax.axis('off')
    
    st.session_state.stplot.pyplot(st.session_state.fig)
    
    df = pd.DataFrame(np.concatenate((tl,tc,tf.expand_dims(100*probs, axis=-1)), axis=-1), columns=['Bit Label','I','Q','Probability (%)'])

    st.session_state.sttable.table(df)
    
    sleep(.2)