# -*- coding: utf-8 -*-
"""
@author: M. S. Neves (msneves@ua.pt)
Instituto de Telecomunicacoes
Universidade de Aveiro
Portugal
"""

import streamlit as st
import matplotlib.pyplot as plt
from fcns.callbacks import learn_button_fcn,stop_button_fcn,reset_button_fcn,choice_M_fcn,snr_update_fcn,choice_const_fcn,choice_demapper_fcn,choice_probs_fcn,md_update_fcn,qb_update_fcn,choice_ps_fcn,choice_mf_fcn,ntaps_update_fcn,bw_update_fcn
from fcns.resetters import reset_epochs
import uuid


def page_setup():
    
    # Set tab title and logo
    st.set_page_config(page_title='Const. Shaping Wizard : IT, Aveiro', layout = 'wide', page_icon = 'logo.ico', initial_sidebar_state='expanded' if st.session_state.get("paused", True) else 'auto')
    st.experimental_set_query_params(embed='true')
    enable_scroll = """
                    <style>
                    .main {
                        overflow: auto;
                    }
                    </style>
                    """
    st.markdown(enable_scroll, unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>Constellation Shaping Wizard</h1>", unsafe_allow_html=True)
    
    # Learning settings 
    if not "learn_button" in st.session_state:
        st.session_state.learning = False
    if not "stop_button" in st.session_state:
        st.session_state.paused = False
        
    
    # Selectboxes
    st.sidebar.title('Constellation Shaping Wizard')
    st.sidebar.subheader('System options')
    st.sidebar.selectbox('Constellation Order',[4,8,16,32,64],key='M',on_change=choice_M_fcn,index=4)
    st.sidebar.slider('SNR [dB]', 0.0, 30.0, key='SNR_dB',on_change=snr_update_fcn,value=18.0,step=0.25)
    st.sidebar.selectbox('Channel constraint',['APC','PPC'],key='norm_mode',on_change=reset_epochs)
    st.sidebar.markdown('Modulation depth')
    st.session_state.md_stplot = st.sidebar.pyplot(st.session_state.get('md_fig',plt.subplots()[0]))
    st.sidebar.slider('Vpp in', 0.1, 1.0, key='md',on_change=md_update_fcn,value=.3,step=.1)
    st.sidebar.slider('Variance of the residual phase noise', 0.0, .05, key='var_phase_noise',on_change=reset_epochs,value=0.,step=.001)
    st.sidebar.slider('Quantization bits (6 for Inf)',1,6,key='qb',on_change=qb_update_fcn,value=6)
    st.sidebar.slider('Normalized Channel Bandwidth',.1,.999,key='bw',on_change=bw_update_fcn,value=.999)
    st.sidebar.subheader('Learning options')
    st.sidebar.selectbox('Probabilistic Shaping',['Don\'t learn','Learn from uniform','Learn from random'],key='choice_probs',on_change=choice_probs_fcn,index=0)
    st.sidebar.selectbox('Geometric Shaping',['Don\'t learn','Learn from square QAM','Learn from random'],key='choice_const',on_change=choice_const_fcn,index=0)
    st.sidebar.slider('Number of Taps in Filters', 1, 63, key='ntaps',on_change=ntaps_update_fcn,value=7,step=2)
    st.sidebar.selectbox('Pulse Shaping',['Use square','Use RRC','Learn from RRC','Learn from random'],key='choice_ps',on_change=choice_ps_fcn,index=0)
    st.sidebar.selectbox('Matched Filter',['Use square','Use RRC','Learn from RRC','Learn from random'],key='choice_mf',on_change=choice_mf_fcn,index=0)
    st.sidebar.selectbox('Demapper',['NN Symwise','NN Bitwise','Min. Dist'],key='choice_demapper',on_change=choice_demapper_fcn,index=2)
    st.sidebar.selectbox('Loss function',['CE','MSE','MI','GMI'],key='loss_fcn', on_change=reset_epochs,index=3)
    
    st.session_state.learn_probs = not st.session_state.choice_probs == 'Don\'t learn'
    st.session_state.learn_const = not st.session_state.choice_const == 'Don\'t learn'
    
    # PLot options
    st.sidebar.subheader('Plot options')
    st.sidebar.checkbox('Show sym. probs. (%)', key='show_probs')
    st.sidebar.checkbox('Show bit labels', key='show_bitlabels')
    
    # Buttons
    st.sidebar.subheader('Control')
    st.sidebar.button('Learn!',on_click=learn_button_fcn,key='learn_button',use_container_width=True,type='primary',disabled=st.session_state.learning)
    st.sidebar.button('Pause Learning',on_click=stop_button_fcn,key='stop_button',use_container_width=True,disabled=not st.session_state.learning)
    st.sidebar.button('Reset Learning',on_click=reset_button_fcn,key='reset_button',use_container_width=True,disabled=st.session_state.learning)
        
        
    
    # Figs and table
    resize = 0.7
    _, plot_area, _ = st.columns((.5, resize / (1 - resize), .5), gap="small")
    st.session_state.fig, st.session_state.ax = plt.subplots()
    st.session_state.fig.patch.set_alpha(0.0)
    st.session_state.ax.axis('off')
    st.session_state.stplot = plot_area.pyplot(st.session_state.fig)
    
    st.session_state.progress = st.progress(0)
    
    resize = 0.7
    _, plot_filters_area, _ = st.columns((.5, resize / (1 - resize), .5), gap="small")
    st.session_state.fig_filters, st.session_state.ax_filters = plt.subplots(1,2)
    st.session_state.fig_filters.patch.set_alpha(0.0)
    st.session_state.ax_filters[0].set_facecolor('black')
    st.session_state.ax_filters[0].xaxis.label.set_color('white')
    st.session_state.ax_filters[0].yaxis.label.set_color('white')
    st.session_state.ax_filters[0].tick_params(axis='x', colors='white')
    st.session_state.ax_filters[0].tick_params(axis='y', colors='white')
    st.session_state.ax_filters[1].set_facecolor('black')
    st.session_state.ax_filters[1].xaxis.label.set_color('white')
    st.session_state.ax_filters[1].yaxis.label.set_color('white')
    st.session_state.ax_filters[1].tick_params(axis='x', colors='white')
    st.session_state.ax_filters[1].tick_params(axis='y', colors='white')
    st.session_state.stplot_filters = plot_filters_area.pyplot(st.session_state.fig_filters)
    
    st.session_state.sttable = st.table()
    
    st.session_state.md_fig, st.session_state.md_ax = plt.subplots(frameon=False,figsize=(3, 1.5))
    md_update_fcn()
        
    if not st.session_state.learning and not st.session_state.paused:
        
        # Attribute random session ID
        st.session_state.session_id = uuid.uuid1()
        
        # ML params
        st.session_state.batch_size = 10000
        st.session_state.num_batches = 100
        st.session_state.num_epochs = 10
        st.session_state.batches_per_plot = 10
        st.session_state.lr_probs = 0
        st.session_state.lr_const = 0
        st.session_state.lr_dmappr = 1e-2
        st.session_state.current_loss = 0
        st.session_state.current_batch = 0
        st.session_state.current_epoch = 0
        
        # set default values
        snr_update_fcn()
        md_update_fcn()
        bw_update_fcn()
        choice_M_fcn()
        choice_probs_fcn()
        choice_const_fcn()
        choice_demapper_fcn()
        choice_ps_fcn()
        choice_mf_fcn()
        
        
    return st.session_state