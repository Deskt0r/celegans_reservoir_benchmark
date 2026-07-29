#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:36:59 2024

@author: felixsr
"""

"""
This script is used to create the configuration files for the experiments.
Configuration files contain:
    'NAME':             The name given to an experiment, printed in terminal when experiment is run.
    'DATASET':          The name referring to a used connectome (lab name + day-in-vitro) or adjacency matrix in general. Printed in terminal when experiment is run and part of the name of files storing the results.
    'RUNS':             For now deprecated, used when multiple runs of the same experiment are desired (will come in handy when doing NULL-models or generating input data on the fly)
    'IN_PATH':          Path to the connectome (not the task dataset of the experiment!)
    'OUT_PATH':         The path that results get stored into.
    'N_PROCESS':        Number of processes for the multiprocessing. Maybe a nice idea to find an optimal value for the computer cluster.
    'ALPHAS':           The scaling factors used for the adjacency matrix.
    'RANDOM_WEIGHTS':   Use weights from adjacency matrix or treat adjacency matrix as binary and generate random weights (unused at moment)
    'LOAD_NODES':       Load input/output nodes from stored file or randomly pick.
    'LOAD_INPUT_DATA':  Load task dataset from stored file or randomly generate.
    'SAVE_RES_STATES':  Save the output of all reservoir nodes (similar to [output_nodes] [n_nodes] = all and now readout layer)
    'METRIC':           Metric used to measure performance of readout module. At the moment correlation coefficient for regression and balanced accuracy for classification. Unfortunately not too many native implementations.
    'TASK'
        'TASK_DOMAIN':  neurogym, reservoirpy or conn2res; Denotes where a task comes from
        'TASK_NAME':    Name of the task (e.g. 'MemoryCapacity')
        'N_TRIALS':     Length of the task dataset, only used when data is randomly generated.
        'INPUT_GAIN':   Scaling factor of the task data. For now I use default values for all tasks.
    'INPUT_NODES':      
        'N_NODES' :     Number of input nodes chosen in the reservoir. task.n_features means that the number of input nodes corresponds to the dimension of the input data.
                        If it is not task.n_features, it has to be an integer!`
        'NODE_SET':     Set from which input nodes are taken. Can only be used if there is a labeling of nodes (e.g. the c.elegans connectome).
        'NODES_FROM':   viable_input_nodes : list of indexes or None
                            List of indexes of possible input nodes to chose from. Length (and indexes) has to correspond to size of conn.w.
                            Can for example be used to make sure that only nodes with outgoing connections are chosen as input nodes.
        'NODES_WITHOUT':excluded_input_nodes : list of indexes or None
                            List of indexes of excluded input nodes. Length (and indexes) has to correspond to size of conn.w.
                            
    'OUTPUT_NODES'
        'N_NODES' :     Number of output nodes chosen in the reservoir. 
                        It has to be an integer!`
        'NODE_SET':     Set from which input nodes are taken. Can only be used if there is a labeling of nodes (e.g. the c.elegans connectome).
        'NODES_FROM':   viable_output_nodes : list of indexes or None
                            List of indexes of possible output nodes to chose from. Length (and indexes) has to correspond to size of conn.w.
        'NODES_WITHOUT':excluded_output_nodes : list of indexes or None
                            List of indexes of excluded output nodes. Length (and indexes) has to correspond to size of conn.w.
    'RESERVOIR'
        'ACT_FCN':      Activation function used inside the reservoir
        'READOUT':      Readout Module used. At the moment the readout module is infered from the task dataset
        'REWIRING':     Allows to rewire the model with internal mechanism in order to create Null-models.
    'W_IN'
        'MODUS'         Chose how the input matrix is set up
                            'uniform' : uniform distribution of weights in [0,1] over all input nodes
                            'boolean' : boolean, all input nodes are "1", otherwise "0"
"""

'''
TODO:
    Add nodes_from and nodes_without
        [(a and not np.all(np.isclose(w[c,:], 0))) for c,a in enumerate(conn.idx_node)]
        AND
        [i for i, x in enumerate(connected_nodes) if x]
        Those should not be called in here, but given as an argument to the create_config_file function!
    Also, create an additional function that generates a file general_exp_config with all information that is the same for all 
    data_name, experiment_name, DIV, ...
'''


import yaml


def _as_plain_int_list(nodes):
    """Convert node index sequences to plain Python ints for YAML-safe dumps."""
    if nodes is None or isinstance(nodes, str):
        return nodes
    return [int(x) for x in nodes]


def create_config_file(
        config_out_path, 
        exp_series,
        data_domain, 
        data_name, 
        experiment_name, 
        DIV,
        task_domain,
        in_path_nodes = None,
        in_path_data = None,
        viable_input_nodes = None,
        excluded_input_nodes = None,
        viable_output_nodes = None,
        excluded_output_nodes = None,
        random_weights = False,
        random_in_weights = True,
        load_nodes = True,
        load_input_data = True,
        save_res_states = True,
        train_readout = True,
        return_res_states= True,
        verbose = False,
        rewiring = False,
        runs=1,
        n_processes=15,
        n_output_nodes = 5,
        n_input_nodes = 'task.n_features'
        ):
    
    config = {}
    
    config['DEFAULT'] = {
                        'NAME':f'{experiment_name}',
                        'DATASET':f'{data_name}_{DIV}',
                        'RUNS':[int(runs)],
                        'IN_PATH': f'./data/{data_domain}/{data_name}/{DIV}/',
                        'OUT_PATH': f'./results/{exp_series}/{data_domain}/{data_name}/{experiment_name}/{DIV}/',
                        'SERIES_CONF_PATH' : f'./experiments/{exp_series}/',
                        'IN_PATH_NODES': in_path_nodes,
                        'IN_PATH_DATA': in_path_data,
                        'N_PROCESS':int(n_processes),
                        #'ALPHAS' : [0.8,0.85,0.91, 0.92, 0.93, 0.9400000000000001, 0.9500000000000001, 0.9600000000000001, 0.9700000000000001, 0.9800000000000001, 0.9900000000000001, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1,1.15,1.2],
                        'ALPHAS' : [0.8,0.85,0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.15,1.2],
                        'RANDOM_WEIGHTS' : random_weights,
                        'RANDOM_IN_WEIGHTS' : random_in_weights,
                        'LOAD_NODES': load_nodes,
                        'LOAD_INPUT_DATA': load_input_data,
                        'SAVE_RES_STATES': save_res_states,
                        'TRAIN_READOUT': train_readout,
                        'RETURN_RES_STATES' : return_res_states,
                        'VERBOSE' : verbose
                        }
    
    if experiment_name in ['MemoryCapacity','mackey_glass','henon_map']:
        config['DEFAULT'].update(   {'METRIC': ['corrcoef','root_mean_squared_error'], # 'corrcoef'
                                     'metric_kwargs' :  {
                                                         'nonnegative': 'absolute'
                                                         }
                                     }
                                 )
    elif experiment_name in ['PerceptualDecisionMaking']:
        config['DEFAULT'].update(   {'METRIC': ['balanced_accuracy_score','filtered_accuracy_score'],
                                     'task_kwargs' : {'cohs': [12.8, 25.6, 51.2]}
                                     }   
            
                                )
    
    elif experiment_name in ['GoNogo']: #PerceptualDecisionMaking
        config['DEFAULT'].update(   {'METRIC': ['balanced_accuracy_score','filtered_accuracy_score']
                                     }
                                 )
        
    config['TASK'] = {
                        'TASK_DOMAIN':f'{task_domain}',
                        'TASK_NAME':f'{experiment_name}',
                        'N_TRIALS':1024,
                        'INPUT_GAIN':1
                        }
    
    config['INPUT_NODES'] = {
                             'N_NODES' : n_input_nodes,
                             'NODE_SET':'random',
                             'NODES_FROM': _as_plain_int_list(viable_input_nodes),
                             'NODES_WITHOUT': _as_plain_int_list(excluded_input_nodes)
                            }
    
    config['OUTPUT_NODES'] = {
                             'N_NODES' : n_output_nodes,
                             'NODE_SET':'random',
                             'NODES_FROM': _as_plain_int_list(viable_output_nodes),
                             'NODES_WITHOUT': _as_plain_int_list(excluded_output_nodes)
                            }
    
    config['RESERVOIR'] = {
                            'ACT_FCN' : 'tanh', # 'sigmoid'
                            'READOUT':'linear_model.ridge',
                            'REWIRING': rewiring
                            }
    
    config['W_IN'] = {
                            'MODUS' : 'uniform',
                            }
    
    with open(config_out_path + f'{data_name}_{experiment_name}_exp_config.yaml', 'w') as file:
        yaml.dump(config, file)

"""
TASK parameters:
    NeuroGym:
        NeuroGym repository:
            dt : 
                step size
            rewards :
                
            timing :
            
        Conn2Res repository:
            n_trials : int, optional
                number of trials to be generated, by default None
            input_gain : float, optional
                gain on the input signal, i.e., scalar multiplier, by default None
            add_bias : bool, optional
                decides whether bias is added to the input signal or not,
                by default False
    
    ReservoirPy:
        ReservoirPy repository:
            -/-
        Conn2Res repository:
            n_trials : int, optional
                number of time steps in input and output, by default None
            horizon : int, numpy.ndarray or list, optional
                shift between input and output, i.e., positive number for
                prediction and negative number for memory task, by default 1
                note that array/list is used for multi-output task
            win : int, optional
                initial window of the input signal to be used for generating the
                delayed output signal in case of memory tasks, by default 30
                note that no values in horizon should exceed this window (in
                absolute value), otherwise ValueError is thrown
            input_gain : float, optional
                gain on the input signal, i.e., scalar multiplier, by default None
            add_bias : bool, optional
                decides whether bias is added to the input signal or not,
                by default False
    
    MemoryCapacity/Conn2Res:
        n_trials : int, optional
            number of time steps in input and output, by default None
        horizon_max : int, optional
            maximum shift between input and output, i.e., negative number 
            for memory capacity task, by default -20
            note that an array of horizons are generated from -1 to 
            inclusive of horizon_max using a step of -1, which 
            defines memory capacity task as a multi-output task, i.e., one 
            task per horizon
        win : int, optional
            initial window of the input signal to be used for generating the
            delayed output signal, by default 30
            note that horizon_max should exceed this window (in
            absolute value), otherwise ValueError is thrown
        low : float, optional
            lower boundary of the output interval of numpy.uniform(),
            by default -1
        high : float, optional
            upper boundary of the output interval of numpy.uniform(),
            by default 1
        input_gain : float, optional
            gain on the input signal, i.e., scalar multiplier, by default None
        add_bias : bool, optional
            decides whether bias is added to the input signal or not,
            by default False
        seed : int, array_like[ints], SeedSequence, BitGenerator, Generator, optional
            seed to initialize the random number generator, by default None
            for details, see numpy.random.default_rng()
    
    PerceptionalDecisionMaking (NeuroGym):
        cohs: list of float, coherence levels controlling the difficulty of
            the task
        sigma: float, input noise level
        dim_ring: int, dimension of ring input and output
    
    GoNogo (NeuroGym):
        -/-
    
    Mackey-Glass (ReservoirPy):
        n_timesteps : int
            Number of timesteps to compute.
        tau : int, default to 17
            Time delay :math:`\\tau` of Mackey-Glass equation.
            By defaults, equals to 17. Other values can
            change the chaotic behaviour of the timeseries.
        a : float, default to 0.2
            :math:`a` parameter of the equation.
        b : float, default to 0.1
            :math:`b` parameter of the equation.
        n : int, default to 10
            :math:`n` parameter of the equation.
        x0 : float, optional, default to 1.2
            Initial condition of the timeseries.
        h : float, default to 1.0
            Time delta between two discrete timesteps.
        seed : int or :py:class:`numpy.random.Generator`, optional
            Random state seed for reproducibility.
    
    Henon-Map (ReservoirPy):
        n_timesteps : int
            Number of timesteps to generate.
        a : float, default to 1.4
            :math:`a` parameter of the system.
        b : float, default to 0.3
            :math:`b` parameter of the system.
        x0 : array-like of shape (2,), default to [0.0, 0.0]
            Initial conditions of the system.
    
    Cart-Pole (Farama Gymnasium):
        
METRIC parameters (objective function):
This shows, which losses have been used in examples (also outside conn2res)
    corrcoef : np.corrcoef
        Pearson's correlation coefficient.
            MemoryCapacity
    balanced_accuracy_score : sklearn.metrics.balanced_accuracy_score
        Balance accuracy score. Good to deal with imbalanced datasets.
            PerceptionalDecisionMaking
    f1_score : sklearn.metrics.f1_score
        F1-score.
            PerceptionalDecisionMaking
    RidgeRegression : 
            Mackey-Glass
    CrossEntropy/Log_Loss : 
            GoNogo
            
AVAILABLE METRICS IN CONN2RES:
    r2 score: Coefficient of determination of the regression R^2.
    mean squared error: 
    root mean squared error:
    mean absolute error:
    corrcoef:
    accuraccy score:
    balanced accurarcy score:
    f1 score:
    precision score:
    recall score:
    
MATCHING TASKS AND METRICS:
    Memory Capacity: corrcoef
    Perceptional Decision Making: Balanced Accuracy Score
    GoNogo: Balanced Accuracy Score (because it is a classification like perceptional decision making?)
    Mackey Glass: corrcoef (because it is a timeseries prediction like memory capacity?)
    Henon Map: corrcoef (because it is a timeseries prediction like memory capacity?)

"""
