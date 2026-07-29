#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 00:23:28 2026

@author: felix
"""

'''
Check if experiments have been successful.
To this end, check if a *score.csv exists and how many non-zero elements *score.csv has.
Collect this in a .csv
'''


import os
import numpy as np
import pandas as pd
import csv

def all_results_close_to_zero(results, tolerance=1e-3, position=1):
    return all(np.isclose(value[position], 0, atol=tolerance) for value in results)

def two_or_more_results_close_to_zero(results, tolerance=1e-6, position=1):
    return sum(np.isclose(value[position], 0, atol=tolerance) for value in results) >= 2



if __name__ == "__main__":
    from pathlib import Path
    TABLES_DIR = Path(__file__).resolve().parent / 'tables'
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    source_connectomes=['celegans']
    culture_list = ['syn_size', 'syn_count', 'phys_contact']
    
    # Specify directories and conditions
    exp_series_raw = ['exp_series_1_celegans','exp_series_1_null_1_celegans','exp_series_1_null_2_celegans','exp_series_1_null_3_celegans',
                   'exp_series_2_celegans','exp_series_2_null_1_celegans','exp_series_2_null_2_celegans','exp_series_2_null_3_celegans',
                   'exp_series_3_celegans','exp_series_3_null_1_celegans','exp_series_3_null_2_celegans','exp_series_3_null_3_celegans',
                   'exp_series_4_celegans','exp_series_4_null_1_celegans','exp_series_4_null_2_celegans','exp_series_4_null_3_celegans',
                   'exp_series_5_celegans','exp_series_5_null_1_celegans','exp_series_5_null_2_celegans','exp_series_5_null_3_celegans'
                   ]
    
    
    experiments = ['MemoryCapacity','PerceptualDecisionMaking','GoNogo','mackey_glass','henon_map']
    
    path_start = './results/'
    
    for culture in culture_list:
        exp_series = [i+'_'+culture for i in exp_series_raw ] 
        csv_file_path = TABLES_DIR / f'validation_results_celegans_{culture}.csv'
        # Create a CSV file to write output
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Series', 'Domain', 'Culture', 'Experiment', 'DIV', 'Status 1'])
        
            for series in exp_series:
                
                if culture == 'syn_count':
                    DIVS_feasible = ['01','02','03','04','05','06','07','08']
                else:
                    DIVS_feasible = ['01','02','03','04','05','06','07']
        
                for experiment in experiments:
                    for DIV in DIVS_feasible:
                        exp_div_path = os.path.join(path_start, series, 'celegans', culture, experiment, str(DIV))
    
                        # Check if path exists
                        if not os.path.exists(exp_div_path):
                            print(f"Path does not exist: {exp_div_path}.")
                            writer.writerow([series, 'celegans', culture, experiment, DIV, 'Path does not exist'])
                            continue
    
                        # Check if score file exists
                        matching_files = []
                        for dirpath, _, filenames in os.walk(exp_div_path):
                            for file in filenames:
                                if file.endswith('scores.csv'):
                                    file_path = os.path.join(dirpath, file)
                                    matching_files.append(file_path)
    
                        if len(matching_files)==0:
                            print(f"No files ending with 'scores.csv' were found in {exp_div_path}.")
                            writer.writerow([series, 'celegans', culture, experiment, DIV, 'Scores do not exist'])
                            continue
    
                        if len(matching_files) < 15:
                            print(f"Less than 15 files ending with 'scores.csv' were found in {exp_div_path}.")
                            writer.writerow([series, 'celegans', culture, experiment, DIV, 'Less than 15 scores found'])
                            continue
    
                        # Collect results
                        all_results = []
                        for file_path in matching_files:
                            if os.path.isfile(file_path):
                                try:
                                    df = pd.read_csv(file_path)
                                    if df.shape[1] < 2:
                                        print(f"Skipped file {file_path}: Not enough columns")
                                        writer.writerow([series, 'celegans', culture, experiment, DIV, 'Not enough columns'])
                                        continue
    
                                    # Collect results
                                    for index, row in df.iterrows():
                                        results = (row.iloc[0], row.iloc[1], row.iloc[2])
                                        all_results.append(results)
    
                                except Exception as e:
                                    print(f"Error opening file {file_path}: {e}")
                                    writer.writerow([series, 'celegans', culture, experiment, DIV, f'Error: {e}'])
    
                        # Check if all collected results are close to zero
                        if all_results_close_to_zero(all_results, position=1) or all_results_close_to_zero(all_results, position=2):
                            writer.writerow([series, 'celegans', culture, experiment, DIV, f'Results are close to zero (tol=1e-3): Results 1: {all_results_close_to_zero(all_results, position=1)}- Results 2: {all_results_close_to_zero(all_results, position=2)}'])
                        elif two_or_more_results_close_to_zero(all_results, position=1) or two_or_more_results_close_to_zero(all_results, position=2):
                            writer.writerow([series, 'celegans', culture, experiment, DIV, f'Two or more results are close to zero (tol=1e-6): Results 1: {two_or_more_results_close_to_zero(all_results, position=1)}- Results 2: {two_or_more_results_close_to_zero(all_results, position=2)}'])
                        else:
                            writer.writerow([series, 'celegans', culture, experiment, DIV, 'All Results seem good'])
