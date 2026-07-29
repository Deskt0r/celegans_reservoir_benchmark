#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 16:35:14 2025

@author: felixsr
"""

'''
This sript is meant to generate all data relevant to a specific experimental configuration
'''

import os
import yaml
from utils.config_writer import create_config_file
from conn2res.connectivity import Conn
import conn2res.tasks
import numpy as np
from utils.data_processing_utils import select_and_save_input_data

'''
This script is used to prepare data for a specific series of experiments. This includes:
    Generate i/o nodes
    Generate i/o datasets
    Generate exp_config files for every in DIV in every data_name
    Save the parameters of the experiment series into a exp_series_config.yaml file
'''

if __name__ == "__main__":
    '''
    IDEA:
    Pick i/o nodes from Sensory/Muscle
    Rewire connectiones only
    In the input matrix w_in, those nodes are set to 1, 0 otherwise
    Readout module gets data from 5 randomly chosen nodes
    '''
    
    exp_series_name = 'exp_series_1_null_1_celegans_phys_contact'
    exp_series_output_path = f'./experiments/{exp_series_name}/'
    if not os.path.exists(exp_series_output_path): 
        os.makedirs(exp_series_output_path) 
        
    # Write exp_series_config file
    series_config = {}
    series_config['DEFAULT'] = {
                                'SERIES_NO' : 'Null_1',
                                'SERIES_NAME' : 'Simple',
                                'SERIES_DESCRIPTION' : 'Random, Boolean I/O Nodes',
                                'IO_NODE_PATH' : f'./experiments/{exp_series_name}/data_domain/data_name/experiment/io-nodes/',
                                'IO_NODE_FUNCTION' : 'Sensory and Muscle'
        }
    with open(os.path.join(exp_series_output_path, f'{exp_series_name}_general_config.yaml'), 'w') as file:
        yaml.dump(series_config, file)
    
    experiments = ['MemoryCapacity','PerceptualDecisionMaking','GoNogo','mackey_glass','henon_map']
    
    # For C Elegans data
    data_domain = 'celegans'
    data_name = 'phys_contact'
    
    # get connected nodes
    # 224 is size of w, conn.w is smaller though
    file_path = f'./data/{data_domain}/{data_name}/01/'
    w = np.load(os.path.join(file_path, 'connectivity.npy'), allow_pickle=True).astype(float)
    conn = Conn(w=w)
    input_nodes = conn.get_nodes(
                                'Sensory',
                                nodes_from = None,
                                nodes_without = None,
                                filename=f'./data/{data_domain}/{data_name}/01/rsn_mapping.npy',
                                n_nodes=None
                                )
    intersect_in = input_nodes
    for age in ['02','03','04','05','06','07']:
        file_path = f'./data/{data_domain}/{data_name}/{age}/'
        w = np.load(os.path.join(file_path, 'connectivity.npy'), allow_pickle=True).astype(float)
        conn = Conn(w=w)
        input_nodes = conn.get_nodes(
                                    'Sensory',
                                    nodes_from = None,
                                    nodes_without = None,
                                    filename=f'./data/{data_domain}/{data_name}/{age}/rsn_mapping.npy',
                                    n_nodes=None
                                    )
        intersect_in = list(set(input_nodes) & set(intersect_in))
    connected_nodes_in = intersect_in
    
    for experiment in experiments:
        for age in ['01','02','03','04','05','06','07']:
            config_output_path = f'./experiments/{exp_series_name}/{data_domain}/{data_name}/{experiment}/{age}/'
            in_path_nodes = f'./experiments/{exp_series_name}/{data_domain}/{data_name}/{experiment}/{age}/io-nodes/'
            in_path_data = f'./experiments/{exp_series_name}/{data_domain}/{data_name}/{experiment}/io-data/'
            if not os.path.exists(config_output_path): 
                os.makedirs(config_output_path)
            if not os.path.exists(in_path_nodes): 
                os.makedirs(in_path_nodes)
            if not os.path.exists(in_path_data): 
                os.makedirs(in_path_data)
            if experiment == 'MemoryCapacity':
                task_domain = 'Conn2ResTask'
                domain = getattr(conn2res.tasks, task_domain)
                task = domain(name=experiment)
                _, _ = task.fetch_data(n_trials=1024, input_gain=1)
            elif experiment == 'PerceptualDecisionMaking' or experiment == 'GoNogo':
                task_domain = 'NeuroGymTask'
                domain = getattr(conn2res.tasks, task_domain)
                task = domain(name=experiment)
                _, _ = task.fetch_data(n_trials=1024, input_gain=1)
            else:
                task_domain = 'ReservoirPyTask'
                domain = getattr(conn2res.tasks, task_domain)
                task = domain(name=experiment)
                _, _ = task.fetch_data(n_trials=1024, input_gain=1)
            create_config_file(
                                config_output_path, 
                                exp_series_name,
                                data_domain, 
                                data_name, 
                                experiment, 
                                age,
                                task_domain,
                                in_path_nodes,
                                in_path_data,
                                viable_input_nodes = None,
                                excluded_input_nodes = None,
                                viable_output_nodes = None,
                                excluded_output_nodes = None,
                                random_weights = False,
                                random_in_weights = False,
                                load_nodes = True,
                                load_input_data = True,
                                save_res_states = False,
                                train_readout = True,
                                return_res_states= True,
                                verbose = False,
                                rewiring = True,
                                runs=1,
                                n_processes=15,
                                n_output_nodes = 5,
                                n_input_nodes = 'task.n_features'
                )
            file_path = f'./data/{data_domain}/{data_name}/{age}/'
            w = np.load(os.path.join(file_path, 'connectivity.npy'), allow_pickle=True).astype(float)
            conn = Conn(w=w)
            
            number=15
            for i in range(number):
                node_path = os.path.join(in_path_nodes,f'{i}/')
                if not os.path.exists(node_path): 
                    os.makedirs(node_path)
                input_nodes = conn.get_nodes(
                                            'Sensory',
                                            nodes_from = connected_nodes_in,
                                            nodes_without = None,
                                            filename=f'./data/{data_domain}/{data_name}/{age}/rsn_mapping.npy',
                                            n_nodes=None
                                            )
                
                input_nodes = conn.get_nodes(
                                            'random',
                                            nodes_from = input_nodes,
                                            nodes_without = None,
                                            filename=None,
                                            n_nodes=task.n_features
                                            )
                output_nodes = conn.get_nodes(
                                            'Muscle',
                                            nodes_from = None,
                                            nodes_without = None,
                                            filename=f'./data/{data_domain}/{data_name}/{age}/rsn_mapping.npy',
                                            n_nodes=None
                                            )
                
                output_nodes = conn.get_nodes(
                                            'random',
                                            nodes_from = output_nodes,
                                            nodes_without = None,
                                            filename=None,
                                            n_nodes=5
                                            )
                
                np.save(os.path.join(node_path, 'input_nodes.npy'), input_nodes)
                np.save(os.path.join(node_path, 'output_nodes.npy'), output_nodes)
            
        with open(os.path.join(config_output_path,f'{data_name}_{experiment}_exp_config.yaml'), 'r') as file:
            exp_config = yaml.safe_load(file)
        select_and_save_input_data(exp_config, 1)