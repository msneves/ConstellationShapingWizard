# -*- coding: utf-8 -*-
"""
@author: M. S. Neves (msneves@ua.pt)
Instituto de Telecomunicacoes
Universidade de Aveiro
Portugal
"""

import numpy as np

def qam(M):
    # Create square QAM constellation
    constellation = np.zeros(M, dtype=complex)
    for i in range(M):
        x = i // np.sqrt(M)
        y = i % np.sqrt(M)
        constellation[i] = (2 * x - np.sqrt(M) + 1 + 1j * (2 * y - np.sqrt(M) + 1)) / np.sqrt(2)

    # Create gray bit labeling
    gray_map = np.zeros(M, dtype=int)
    gray_map[0] = 0
    for i in range(1, M):
        col = i // int(np.sqrt(M))
        row = i % int(np.sqrt(M))
        gray_map[i] = (row >> 1) ^ row + ((col >> 1) ^ col) * int(np.sqrt(M))

    constmp = constellation.copy()
    try:
        constmp[gray_map] = constellation
    except:
        pass # possibly gives error if non square QAM   
    constellation = constmp
    
    constellation = np.column_stack((constellation.real, constellation.imag))

    return constellation