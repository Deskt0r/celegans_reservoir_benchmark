#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 18:28:03 2025

@author: felixsr
"""

import argparse
import os
import pickle
from conn_pipeline import pipeline
from utils.result_processing_utils import ext_cond_with_alpha

'''
This script is used to start an experiment series from a slurm job script.
Keywords include:
    path to f'./experiments/experiments_series_{}/data_domain/data_name'
From there, exp_config files are read, which contain all other necessary arguments
'''

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                        prog='Main Pipeline',
                        description='This script is used to start an experiment series for a specific set of connectivity matrixes from a slurm job script.',
                        epilog='gl hf')
    
    parser.add_argument('--path_exp', 
                        help='provide path to ./experiments/experiments_series_{}/data_domain/data_name',
                        default='./experiments/exp_series_1_celegans_syn_count/celegans/syn_count/')
    
    parser.add_argument('--path_res', 
                        help='provide path to ./results/experiments_series_{}/data_domain/data_name',
                        default='./results/exp_series_1_celegans_syn_count/celegans/syn_count/')
    
    parser.add_argument('--excl_DIV',
                        nargs='*',
                        type=str,
                        default=[],
                        help='list of excluded DIVs (for example because of too small graphs)',
                        )
    
    args = parser.parse_args()
    
    print(args.path_exp)
    print(args.path_res)
    
    path_exp = args.path_exp
    path_res = args.path_res
    list_excl = args.excl_DIV
    
    lead,data_name = os.path.split(path_exp)
    if not data_name:
        _, data_name = os.path.split(lead)
    
    all_rs_cond = {}
    list_DIVs = []
    list_experiments = []
    for i in os.listdir(path_exp):
        path_to_experiment = os.path.join(path_exp,i)
        list_experiments.append(i)
        for j in os.listdir(os.path.join(path_exp,i)):
            # Make sure only directories that correspond to DIVs (no e.g. io_nodes) are traversed
            if j.isdigit() and j not in list_excl:
                path_to_DIV = os.path.join(path_to_experiment,j)
                if j not in list_DIVs:
                    list_DIVs.append(j)
                for m in os.listdir(path_to_DIV):
                    if m.endswith('config.yaml'):
                        path_to_config = os.path.join(path_to_DIV,m)
                        print(path_to_config)
                results = pipeline(path_to_config)
                for res in results:
                    try:
                        rs_cond = ext_cond_with_alpha(res)
                        all_rs_cond.update(rs_cond)
                    except: # numpy.linalg.LinAlgError: cond is not defined on empty arrays
                        print('All zero matrix for:',m)
    if not os.path.exists(path_res): 
        os.makedirs(path_res) 
    with open(os.path.join(path_res, 'all_rs_cond.pckl'), "wb") as data:
        pickle.dump(all_rs_cond, data)
