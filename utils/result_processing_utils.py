#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 18:26:15 2025

@author: felixsr
"""

import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import normalize

def read_max_performances(file_path):
    # in case 2 performance metrics are used
    try:
        df = pd.read_csv(file_path).fillna(0)
        if df.shape[1] < 2:
            # Ensure the dataframe has at least two columns
            print(f"Skipped file {file_path}: Not enough columns")
            return None
        row_max_1 = df.iloc[df.iloc[:, 1].idxmax()]
        row_max_2 = df.iloc[df.iloc[:, 2].idxmax()]
        return row_max_1.iloc[0], row_max_1.iloc[1], row_max_2.iloc[0], row_max_2.iloc[2]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    
def ext_cond_with_alpha(res):
    rs_all_dict_cond = {}
    for key in list(res.keys()):
        res_states_test = res[key]['test']
        res_states_test = np.stack(res_states_test,axis=0)
        # Delete input nodes from reservoir states
        t2=int(key[2].rfind(']'))
        input_nodes = [int(s) for s in re.findall(r'\b\d+\b', key[2][:t2])]
        res_states_test = np.delete(res_states_test, input_nodes, axis=-1)
        # For [n_sequence, length_sequence, n_nodes]: Turn into [n_sequence, length_sequence * n_nodes]
        if len(res_states_test.shape) >= 2:
            res_states_test = np.vstack(res_states_test)
        # Normalize
        res_states_test = normalize(res_states_test,axis=1)
        # Delete all zero rows: Those were never reached by the input signal
        zero_col = np.where(np.all(np.isclose(res_states_test, 0), axis=0))
        res_states_test = np.delete(res_states_test, zero_col, axis=-1)
        # Calculate condition number
        rs_all_dict_cond[key] = np.linalg.cond(np.matmul(res_states_test.T,res_states_test))
    return rs_all_dict_cond
