#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 21:12:08 2026

@author: felix
"""

import os
import csv
from utils.result_processing_utils import read_max_performances


def retrieve_data(list_experiments, list_DIVs, path_res):
    '''
    

    Parameters
    ----------
    list_experiments : TYPE
        DESCRIPTION.
    list_DIVs : TYPE
        DESCRIPTION.
    path_res : TYPE
        DESCRIPTION.

    Returns
    -------
    data_dict : dictionary
        Should contain the max performance (alpha, value) for every experiment for every DIV FOR ONE CULTURE.

    '''
    data_dict = {}
    for j in list_experiments:
        data_dict[j] = {}
        for counter, DIV in enumerate(list_DIVs):
            data_dict[j][DIV] = []
            exp_div_path = os.path.join(path_res,j,str(DIV))
            if not os.path.exists(exp_div_path):
                print(f"Path does not exist: {exp_div_path}")
                continue
            for root, dirs, files in os.walk(exp_div_path):
                for file in files:
                    if file.endswith('scores.csv'):
                        file_path = os.path.join(root, file)
                        result = read_max_performances(file_path)
                        if result:
                            data_dict[j][DIV].append({'alpha_1': result[0], 'value_1': result[1], 'alpha_2': result[2], 'value_2': result[3]})
        
    return data_dict

if __name__ == "__main__":
    from pathlib import Path
    TABLES_DIR = Path(__file__).resolve().parent / 'tables'
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    list_experiments = ['MemoryCapacity','PerceptualDecisionMaking','GoNogo','mackey_glass','henon_map']
    list_series_raw = ['exp_series_1_celegans','exp_series_1_null_1_celegans','exp_series_1_null_2_celegans','exp_series_1_null_3_celegans',
                   'exp_series_2_celegans','exp_series_2_null_1_celegans','exp_series_2_null_2_celegans','exp_series_2_null_3_celegans',
                   'exp_series_3_celegans','exp_series_3_null_1_celegans','exp_series_3_null_2_celegans','exp_series_3_null_3_celegans',
                   'exp_series_4_celegans','exp_series_4_null_1_celegans','exp_series_4_null_2_celegans','exp_series_4_null_3_celegans',
                   'exp_series_5_celegans','exp_series_5_null_1_celegans','exp_series_5_null_2_celegans','exp_series_5_null_3_celegans'
                   ]
    
    list_cultures = ['syn_size', 'syn_count', 'phys_contact']
    data_domain='celegans'
                   
    for culture in list_cultures:
        if culture=='syn_count':
            list_ages = ['01','02','03','04','05','06','07','08']
        else:
            list_ages = ['01','02','03','04','05','06','07']
        list_series = [i+'_'+ culture for i in list_series_raw ]
        file_path = TABLES_DIR / f'results_celegans_{culture}.csv'
        # Create a CSV file to write output
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Series', 'Domain', 'Culture', 'Experiment', 'Age', 'Run', 'Alpha 1', 'Metric 1', 'Alpha 2', 'Metric 2'])
            for series in list_series:
                path_res = f'./results/{series}/celegans/{culture}/'
                data_dict = retrieve_data(list_experiments, list_ages, path_res)
    
                for experiment in data_dict.keys():
                    for age in data_dict[experiment].keys():
                        data = data_dict[experiment][age]
                        # For syn_size/phys_contact, developmental age folder 07 is labeled as age 08
                        # in the CSV for alignment with syn_count labeling in downstream analysis.
                        if culture!='syn_count':
                            if age=='07':
                                age='08'
                        for i in range(len(data)):
                            writer.writerow([series, data_domain, culture, experiment, age, i, data[i]['alpha_1'], data[i]['value_1'], data[i]['alpha_2'], data[i]['value_2']])
