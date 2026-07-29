#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 18:27:44 2025

@author: felixsr
"""

import yaml
import pickle
import os
import time
import multiprocessing as mp

import numpy as np
import pandas as pd

import conn2res.tasks
from conn2res.connectivity import Conn
from conn2res.reservoir import EchoStateNetwork
from conn2res.readout import Readout
from conn2res import readout, plotting
from utils.data_processing_utils import select_nodes, global_curveball_directed

SEED = 50
RNG = np.random.default_rng(seed=SEED)

def single_run(
    w, x, y, task, path_run, exp_config, node_dir = None, rewire=False, filename=None, data_ident = None
):
    if isinstance(exp_config['DEFAULT']['METRIC'],str):
        # If more than one metric is used (e.g. f1 score and balanced accurarcy score)
        if exp_config['DEFAULT']['METRIC'].startswith("[") and exp_config['DEFAULT']['METRIC'].endswith("]"):
            # Remove the enclosing square brackets
            inner_content = exp_config['DEFAULT']['METRIC'][1:-1]
            # Split by comma and strip whitespace from each element
            metric = [elem.strip() for elem in inner_content.split(",")]
        else:
            metric = exp_config['DEFAULT']['METRIC']
    else:
        metric = exp_config['DEFAULT']['METRIC']  
    # Initialize connectome (i.e. the graph inside the reservoir) from adjacency matrix w
    conn = Conn(w=w)
    
    # Rewiring to create Null model
    if rewire:
        conn.w = global_curveball_directed(conn.w.copy())
        
    if exp_config['DEFAULT']['RANDOM_WEIGHTS'] == True: 
        i_a, i_b = np.nonzero(w)
        for index in zip(i_a, i_b):
            w[index[0],index[1]]=RNG.normal()
             
    # Scaling of the reservoir's network
    conn.scale_and_normalize()
    # Load input and output nodes
    if exp_config['DEFAULT']['LOAD_NODES'] == True:
        input_nodes = np.load(os.path.join(node_dir, 'input_nodes.npy'))
        input_nodes = [n-np.count_nonzero(conn.idx_node[:n]==False) for n in input_nodes]
        output_nodes = np.load(os.path.join(node_dir, 'output_nodes.npy'))
        output_nodes = [n-np.count_nonzero(conn.idx_node[:n]==False) for n in output_nodes]
    else:
        path_to_series_config = exp_config['DEFAULT']['SERIES_CONF_PATH']
        for file in os.listdir(path_to_series_config):
            if file.endswith('general_config.yaml'):
                path_to_series_config = os.path.join(exp_config['DEFAULT']['SERIES_CONF_PATH'],file)
        with open(path_to_series_config, 'r') as file:
            exp_series_config = yaml.safe_load(file)
        if exp_series_config['DEFAULT']['SERIES_NAME'] == 'DIV Nodes':
            connected_nodes = [(a and not np.all(np.isclose(w[c,:], 0))) for c,a in enumerate(conn.idx_node)]
            connected_nodes = [i for i, x in enumerate(connected_nodes) if x]
            exp_config['INPUT_NODES']['NODES_FROM'] = connected_nodes
        elif exp_series_config['DEFAULT']['SERIES_NAME'] == 'DIV Nodes, Uniform Weights':
            connected_nodes = [a for a in conn.idx_node]
            connected_nodes = [i for i, x in enumerate(connected_nodes) if x]
            exp_config['INPUT_NODES']['NODES_FROM'] = connected_nodes
            exp_config['OUTPUT_NODES']['NODES_FROM'] = connected_nodes
        input_nodes, output_nodes = select_nodes(conn, task, exp_config)
        input_nodes = [n-np.count_nonzero(conn.idx_node[:n]==False) for n in input_nodes]
        output_nodes = [n-np.count_nonzero(conn.idx_node[:n]==False) for n in output_nodes]
    
    # Generate "true" node_ident, which takes "shift" into account
    node_ident = '-'.join([str(i) for i in output_nodes])
    node_ident = str(input_nodes) + '_' + node_ident 
    
    # Initialize the matrix that is used to feed the input data to the reservoir
    w_in = np.zeros((len(input_nodes), conn.n_nodes))
    if exp_config['DEFAULT']['RANDOM_IN_WEIGHTS']:
        w_in[:, input_nodes] = RNG.uniform(-1,1,size=(len(input_nodes),len(input_nodes)))
    else:
        try:
            w_in[:, input_nodes] = np.eye(len(input_nodes))
        except:
            print('trouble with',exp_config['DEFAULT']['NAME'], exp_config['DEFAULT']['DATASET'])
            print('input nodes',input_nodes)
    # Load reservoir
    esn = EchoStateNetwork(w=conn.w, activation_function='tanh')
    
    # Load readout module, it is selected based on the ground truth data
    readout_module = Readout(estimator=readout.select_model(y))

    x_train, x_test, y_train, y_test = readout.train_test_split(x, y)
    
    # Load alpha values, the scaling factor of the network weights
    df_alpha = []
    ALPHAS = exp_config['DEFAULT']['ALPHAS'] 
    ALPHAS = [float(alpha) for alpha in ALPHAS]
    
    if exp_config['DEFAULT']['RETURN_RES_STATES'] == True:
        rs_all_dict = {}
    for alpha in ALPHAS:

        # scale network weights
        esn.w = alpha * conn.w

        # feed data to reservoir and records reservoir states of output nodes
        rs_train = esn.simulate(
            ext_input=x_train, w_in=w_in,
            output_nodes=output_nodes
        )

        rs_test = esn.simulate(
            ext_input=x_test, w_in=w_in,
            output_nodes=output_nodes
        )
        
        if exp_config['DEFAULT']['SAVE_RES_STATES'] == True or exp_config['DEFAULT']['RETURN_RES_STATES'] == True:
            rs_all_train = esn.simulate(
                ext_input=x_train, w_in=w_in
            )

            rs_all_test = esn.simulate(
                ext_input=x_test, w_in=w_in
            )
            
        # fit readout module to outpt nodes states, hand of additional arguments for the metric if available
        if exp_config['DEFAULT']['TRAIN_READOUT'] == True:
            if 'metric_kwargs' in exp_config['DEFAULT'].keys():
                df_res = readout_module.run_task(
                    X=(rs_train, rs_test), y=(y_train, y_test),
                    sample_weight=None,
                    metric=metric,
                    readout_modules=None, readout_nodes=None,
                    **exp_config['DEFAULT']['metric_kwargs']
                )
            else:
                df_res = readout_module.run_task(
                    X=(rs_train, rs_test), y=(y_train, y_test),
                    sample_weight=None,
                    metric=metric,
                    readout_modules=None, readout_nodes=None
                )
                
            print(df_res)
            # store performance values in df_alpha
            df_res['alpha'] = np.round(alpha, 3)
            df_alpha.append(df_res)
        
        # for a certain range of alpha values, generate plots: reservoir states and readout module output
        if (0.8 <= alpha <= 1.2):
            path_alpha = path_run + f'{alpha}/'
            if not os.path.exists(path_alpha): 
                os.makedirs(path_alpha) 
                
            if exp_config['DEFAULT']['VERBOSE'] == True:
                if exp_config['DEFAULT']['SAVE_RES_STATES'] == True:
                    np.savez_compressed(os.path.join(path_alpha , f'res_states_train_{task.name}_{alpha}.npy'), rs_all_train)
                    np.savez_compressed(os.path.join(path_alpha , f'res_states_test_{task.name}_{alpha}.npy'), rs_all_test)
                    plotting.plot_reservoir_states(
                                x=x_test, reservoir_states=rs_all_test,
                                title=f'{task.name} - training',
                                rc_params={'figure.dpi': 300, 'savefig.dpi': 300},
                                show=False,
                                savefig=True,
                                fname=os.path.join(path_alpha , f'res_states_{task.name}_{alpha}')
                            )
                else:
                    plotting.plot_reservoir_states(
                                x=x_test, reservoir_states=rs_test,
                                title=f'{task.name} - training',
                                rc_params={'figure.dpi': 300, 'savefig.dpi': 300},
                                show=False,
                                savefig=True,
                                fname=os.path.join(path_alpha , f'res_states_{task.name}_{alpha}')
                            )
                
                if exp_config['DEFAULT']['TRAIN_READOUT'] == True:
                    if hasattr(readout_module, 'decision_function'):
                        plotting.plot_diagnostics(
                                    x=x_test, y=y_test, reservoir_states=rs_test,
                                    trained_model=readout_module.model, title=f'{task.name} - testing',
                                    savefig=True,
                                    fname=os.path.join(path_alpha, f'performance_test_{task.name}_{alpha}'),
                                    rc_params={'figure.dpi': 300, 'savefig.dpi': 300},
                                    show=False
                                )
                        plotting.plot_diagnostics(
                                    x=x_train, y=y_train, reservoir_states=rs_train,
                                    trained_model=readout_module.model, title=f'{task.name} - training',
                                    savefig=True,
                                    fname=os.path.join(path_alpha, f'performance_train_{task.name}_{alpha}'),
                                    rc_params={'figure.dpi': 300, 'savefig.dpi': 300},
                                    show=False
                                )
            if exp_config['DEFAULT']['RETURN_RES_STATES'] == True:
                rs_all_dict[(exp_config['DEFAULT']['NAME'],str(exp_config['DEFAULT']['DATASET'].split("_")[-1]),node_ident,data_ident,str(alpha))] = {'train':rs_all_train,'test':rs_all_test}
    
    if exp_config['DEFAULT']['TRAIN_READOUT'] == True:
        df_alpha = pd.concat(df_alpha, ignore_index=True)
        # df_alpha is altered if more than one metric is used
        if isinstance(metric, list):
            columns = metric.copy()
            columns.insert(0, 'alpha')
            df_alpha = df_alpha[columns]
        else:
            df_alpha = df_alpha[['alpha', metric]]
        
        # for NaN values in pearsons correlation coefficient
        df_alpha.fillna(0)
        # save performance values to csv
        df_alpha.to_csv(
            os.path.join(path_run, f'{filename}_scores.csv'),
            index=False
            )
    if exp_config['DEFAULT']['RETURN_RES_STATES'] == True:
        return rs_all_dict

def pipeline(path_to_config):
    
    # Open up configuration file, this contains all necessary information for an experiment
    with open(path_to_config, 'r') as file:
        exp_config = yaml.safe_load(file)
    print('Running experiment',exp_config['DEFAULT']['NAME'],'for the data set',exp_config['DEFAULT']['DATASET'])
    print(exp_config['TASK']['TASK_NAME'])
    
    # Instantiate task object
    domain = getattr(conn2res.tasks, exp_config['TASK']['TASK_DOMAIN'])
    task = domain(name=exp_config['TASK']['TASK_NAME'])
    
    # Load adjacency matrix
    w = np.load(os.path.join(exp_config['DEFAULT']['IN_PATH'], 'connectivity.npy'), allow_pickle=True).astype(float)
    
    # Connectome is used for file names, it is not the actual matrix
    connectome = exp_config['DEFAULT']['DATASET']
    
    # Null models with loaded nodes
    if exp_config['RESERVOIR']['REWIRING']==True and exp_config['DEFAULT']['LOAD_NODES']==True:
        params = []
        
        # data and nodes have to be prepared in advance
        data_path = exp_config['DEFAULT']['IN_PATH_DATA']
        node_path = exp_config['DEFAULT']['IN_PATH_NODES']
        
        # iterate through all folders containing i/o nodes
        for n in os.listdir(node_path):
            node_dir = os.path.join(node_path,n) 
            # iterate through all folders containing i/o data
            for m in os.listdir(data_path):
                # path to one i/o dataset
                data_dir = os.path.join(data_path,m)
                # Run Null model for RUNS times, each time rewiring the connectivity matrix and choosing random i/o nodes
                for i in range(int(exp_config['DEFAULT']['RUNS'][0])):
                    # path where output is stored (/DIV/node/data/)
                    path_run = os.path.join(os.path.join(exp_config['DEFAULT']['OUT_PATH'],f'{os.path.basename(node_dir)}/'),f'{m}/')
                    if not os.path.exists(path_run): 
                        os.makedirs(path_run) 
                    
                    # use pickle to conserve lists
                    with open(os.path.join(data_dir, 'input.pckl'), "rb") as input_data:
                        x = pickle.load(input_data)
                    with open(os.path.join(data_dir, 'output.pckl'), "rb") as input_data:
                        y = pickle.load(input_data)
                    
                    # n_features attribute is set when calling .fetch_data. So it has to be manually set when loading data
                    if exp_config['TASK']['TASK_DOMAIN'] == 'NeuroGymTask':
                        if x[0].squeeze().ndim == 1:
                            setattr(task, 'n_features', 1)
                        elif x[0].squeeze().ndim == 2:
                            setattr(task, 'n_features', x[0].shape[1])
                
                        if y[0].squeeze().ndim == 1:
                            setattr(task, 'n_targets', 1)
                        elif y[0].squeeze().ndim == 2:
                            setattr(task, 'n_targets', y[0].shape[1])
                    else:
                        if x.squeeze().ndim == 1:
                            setattr(task, 'n_features', 1)
                        elif x.squeeze().ndim == 2:
                            setattr(task, 'n_features', x.shape[1])
        
                        if y.squeeze().ndim == 1:
                            setattr(task, 'n_targets', 1)
                        elif y.squeeze().ndim == 2:
                            setattr(task, 'n_targets', y.shape[1])
                    
                    # repetitions for more input nodes
                    if exp_config['DEFAULT']['RANDOM_IN_WEIGHTS']:
                        repetitions = int(exp_config['INPUT_NODES']['N_NODES'] / task.n_features)
                        if isinstance(x,list):
                            x = [np.tile(i,(1,repetitions)) for i in x]
                        else:
                            x = np.tile(x,(1,repetitions))
                            
                    params.append(
                        {
                            'w': w.copy(),
                            'x': x,
                            'y': y,
                            'task': task,
                            'path_run': path_run,
                            'exp_config':exp_config,
                            'node_dir' : node_dir,
                            'rewire':True,
                            'filename': f'{connectome}_null_{i}',
                            'data_ident': m
                        }
                    )
            
        print('Params:',len(params))
        print('\nINITIATING PROCESSING TIME')
        t0 = time.perf_counter()
        
        pool = mp.Pool(processes=exp_config['DEFAULT']['N_PROCESS'])
        res = [pool.apply_async(single_run, (), p) for p in params]
        results = [r.get() for r in res]
        pool.close()
        
        print('\nTOTAL PROCESSING TIME')
        print(time.perf_counter()-t0, "seconds process time")
        print('END')
        return results
        
        
    # NULL MODELS with node choice at runtime
    if exp_config['RESERVOIR']['REWIRING']==True:
        params = []
        
        # data and nodes have to be prepared in advance
        data_path = exp_config['DEFAULT']['IN_PATH_DATA']
        
        # iterate through all folders containing i/o data
        for m in os.listdir(data_path):
            # path to one i/o dataset
            data_dir = os.path.join(data_path,m)
            # Run Null model for RUNS times, each time rewiring the connectivity matrix and choosing random i/o nodes
            for i in range(int(exp_config['DEFAULT']['RUNS'][0])):
                # path where output is stored (/DIV/run/data/)
                path_run = os.path.join(os.path.join(exp_config['DEFAULT']['OUT_PATH'],f'{i}/'),f'{m}/')
                if not os.path.exists(path_run): 
                    os.makedirs(path_run)
                
                # use pickle to conserve lists
                with open(os.path.join(data_dir, 'input.pckl'), "rb") as input_data:
                    x = pickle.load(input_data)
                with open(os.path.join(data_dir, 'output.pckl'), "rb") as input_data:
                    y = pickle.load(input_data)
                
                # n_features attribute is set when calling .fetch_data. So it has to be manually set when loading data
                if exp_config['TASK']['TASK_DOMAIN'] == 'NeuroGymTask':
                    if x[0].squeeze().ndim == 1:
                        setattr(task, 'n_features', 1)
                    elif x[0].squeeze().ndim == 2:
                        setattr(task, 'n_features', x[0].shape[1])
            
                    if y[0].squeeze().ndim == 1:
                        setattr(task, 'n_targets', 1)
                    elif y[0].squeeze().ndim == 2:
                        setattr(task, 'n_targets', y[0].shape[1])
                else:
                    if x.squeeze().ndim == 1:
                        setattr(task, 'n_features', 1)
                    elif x.squeeze().ndim == 2:
                        setattr(task, 'n_features', x.shape[1])
    
                    if y.squeeze().ndim == 1:
                        setattr(task, 'n_targets', 1)
                    elif y.squeeze().ndim == 2:
                        setattr(task, 'n_targets', y.shape[1])
                
                # repetitions for more input nodes
                if exp_config['DEFAULT']['RANDOM_IN_WEIGHTS']:
                    repetitions = int(exp_config['INPUT_NODES']['N_NODES'] / task.n_features)
                    if isinstance(x,list):
                        x = [np.tile(i,(1,repetitions)) for i in x]
                    else:
                        x = np.tile(x,(1,repetitions))
                
                params.append(
                    {
                        'w': w.copy(),
                        'x': x,
                        'y': y,
                        'task': task,
                        'path_run': path_run,
                        'exp_config':exp_config,
                        'rewire':True,
                        'filename': f'{connectome}_null_{i}',
                        'data_ident': m
                    }
                )
            
        print('Params:',len(params))
        print('\nINITIATING PROCESSING TIME')
        t0 = time.perf_counter()
        
        pool = mp.Pool(processes=exp_config['DEFAULT']['N_PROCESS'])
        res = [pool.apply_async(single_run, (), p) for p in params]
        results = [r.get() for r in res]
        pool.close()
        
        print('\nTOTAL PROCESSING TIME')
        print(time.perf_counter()-t0, "seconds process time")
        print('END')
        return results
    elif (exp_config['RESERVOIR']['REWIRING']==False):
        params = []
        
        # data and nodes have to be prepared in advance
        data_path = exp_config['DEFAULT']['IN_PATH_DATA']
        node_path = exp_config['DEFAULT']['IN_PATH_NODES']
        
        # iterate through all folders containing i/o nodes
        for n in os.listdir(node_path):
            node_dir = os.path.join(node_path,n) 
            # iterate through all folders containing i/o data
            for m in os.listdir(data_path):
                # path to one i/o dataset
                data_dir = os.path.join(data_path,m)
                # path where output is stored (/DIV/node/data/)
                path_run = os.path.join(os.path.join(exp_config['DEFAULT']['OUT_PATH'],f'{os.path.basename(node_dir)}/'),f'{m}/')
                if not os.path.exists(path_run): 
                    os.makedirs(path_run) 
                    
                # use pickle to conserve lists
                with open(os.path.join(data_dir, 'input.pckl'), "rb") as input_data:
                    x = pickle.load(input_data)
                with open(os.path.join(data_dir, 'output.pckl'), "rb") as input_data:
                    y = pickle.load(input_data)
                
                # n_features attribute is set when calling .fetch_data. So it has to be manually set when loading data
                if exp_config['TASK']['TASK_DOMAIN'] == 'NeuroGymTask':
                    if x[0].squeeze().ndim == 1:
                        setattr(task, 'n_features', 1)
                    elif x[0].squeeze().ndim == 2:
                        setattr(task, 'n_features', x[0].shape[1])
            
                    if y[0].squeeze().ndim == 1:
                        setattr(task, 'n_targets', 1)
                    elif y[0].squeeze().ndim == 2:
                        setattr(task, 'n_targets', y[0].shape[1])
                else:
                    if x.squeeze().ndim == 1:
                        setattr(task, 'n_features', 1)
                    elif x.squeeze().ndim == 2:
                        setattr(task, 'n_features', x.shape[1])
    
                    if y.squeeze().ndim == 1:
                        setattr(task, 'n_targets', 1)
                    elif y.squeeze().ndim == 2:
                        setattr(task, 'n_targets', y.shape[1])
                        
                # repetitions for more input nodes
                if exp_config['DEFAULT']['RANDOM_IN_WEIGHTS']:
                    repetitions = int(exp_config['INPUT_NODES']['N_NODES'] / task.n_features)
                    if isinstance(x,list):
                        x = [np.tile(i,(1,repetitions)) for i in x]
                    else:
                        x = np.tile(x,(1,repetitions))
                
                params.append(
                    {
                        'w': w.copy(),
                        'x': x,
                        'y': y,
                        'task': task,
                        'path_run': path_run,
                        'exp_config':exp_config,
                        'node_dir' : node_dir,
                        'rewire':False,
                        'filename': f'{connectome}_empirical',
                        'data_ident': m
                    }
                )
            
        print('Params:',len(params))
        print('\nINITIATING PROCESSING TIME')
        t0 = time.perf_counter()
        
        pool = mp.Pool(processes=exp_config['DEFAULT']['N_PROCESS'])
        res = [pool.apply_async(single_run, (), p) for p in params]
        results = [r.get() for r in res]
        pool.close()
        
        print('\nTOTAL PROCESSING TIME')
        print(time.perf_counter()-t0, "seconds process time")
        print('END')
        return results