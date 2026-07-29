#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 18:25:24 2025

@author: felixsr
"""

from conn2res.connectivity import Conn
import conn2res.tasks
import numpy as np
import os
import pickle

def select_nodes(conn, task, exp_config):
    '''
    Based on the parameters set in the configuration file exp_config, certain nodes are chosen as input and output nodes
    In the first step, a pre selection is made and in the second step, a random sample might be drawn.

    Parameters
    ----------
    conn : class from conn2res.connectivity
        "Class that represents a weighted or unweighted network using connectivity data" from conn2res.
    task : class from conn2res.tasks
        Class that represents the chosen task, task.n_features is the number of features that a dataset corresponding to a task has.
    exp_config : dictionary
        Loaded configuration file, defining parameters

    Returns
    -------
    input_nodes, output_nodes : lists
        lists of indexes of input and output nodes.
        CAUTION: Indexes correspond to "full" connectome, i.e. NOT the larges connected component.
        This means that the indes might need to be corrected when used with conn.w ([n-np.count_nonzero(conn.idx_node[:n]==False) for n in input_nodes])
    '''
    if exp_config['INPUT_NODES']['N_NODES'] == 'task.n_features':
        exp_config['INPUT_NODES']['N_NODES'] = getattr(task, exp_config['INPUT_NODES']['N_NODES'][5:])
    input_nodes = conn.get_nodes(
                                exp_config['INPUT_NODES']['NODE_SET'],
                                nodes_from = exp_config['INPUT_NODES']['NODES_FROM'],
                                nodes_without = exp_config['INPUT_NODES']['NODES_WITHOUT'],
                                filename=os.path.join(exp_config['DEFAULT']['IN_PATH'],'rsn_mapping.npy'),
                                n_nodes=int(exp_config['INPUT_NODES']['N_NODES'])
                                )
    # OUTPUT nodes
    output_nodes = conn.get_nodes(
                                exp_config['OUTPUT_NODES']['NODE_SET'],
                                nodes_from = exp_config['OUTPUT_NODES']['NODES_FROM'],
                                nodes_without = exp_config['OUTPUT_NODES']['NODES_WITHOUT'],
                                filename=os.path.join(exp_config['DEFAULT']['IN_PATH'],'rsn_mapping.npy'),
                                n_nodes=int(exp_config['OUTPUT_NODES']['N_NODES'])
                                )
    return input_nodes, output_nodes

def select_and_save_input_data(exp_config, number, repetitions=None, special_path=None):
    '''
    Select data to use as input/output ('task') dataset from available conn2res tasks

    Parameters
    ----------
    exp_config : dictionary
        Loaded configuration file, defining parameters
    number : int
        Number of i/o datasets that are generated and saved
    task : string
        -
    repetitions : int
    
    special_path : string
    
    Returns
    -------
    -
    '''
    
    domain = getattr(conn2res.tasks, exp_config['TASK']['TASK_DOMAIN'])
    if 'task_kwargs' in exp_config['TASK'].keys():
        task = domain(name=exp_config['TASK']['TASK_NAME'],**exp_config['TASK']['task_kwargs'])
        print(task.cohs)
    else:
        task = domain(name=exp_config['TASK']['TASK_NAME'])
                      
    for i in range(number):
        x, y = task.fetch_data(n_trials=int(exp_config['TASK']['N_TRIALS']), input_gain=int(exp_config['TASK']['INPUT_GAIN']) )
        if special_path:
            path = special_path
        else:
            path = os.path.join(exp_config['DEFAULT']['IN_PATH_DATA'],f'{number}')
        if not os.path.exists(path): 
            os.makedirs(path)
        if repetitions:
            if x is list:
                x = [np.tile(i,(1,repetitions)) for i in x]
            else:
                x = np.tile(x,(1,repetitions))
            
        # Use pickle to preserve lists
        with open(os.path.join(path, 'input.pckl'), "wb") as input_data:
            pickle.dump(x, input_data)
        with open(os.path.join(path, 'output.pckl'), "wb") as output_data:
            pickle.dump(y, output_data)

def validate_adjacency_matrix(w):
    if not isinstance(w, np.ndarray):
        raise TypeError("The matrix should be a numpy ndarray.")
    
    if w.shape[0] != w.shape[1]:
        raise ValueError("The matrix should be a square matrix.")
    
    if not np.all(np.isreal(w)):
        raise ValueError("The matrix should have real number entries.")
        
def test_curveball_correctness(original_w, shuffled_w):
    """Test the correctness of the Curveball algorithm on weighted adjacency matrix."""
    # Ensure correct matrix shape
    assert original_w.shape == shuffled_w.shape, "Shape of adjacency matrices must remain the same."
    
    # Check the in-degree and out-degree sequences are unchanged
    original_out_degree = np.sum(original_w != 0, axis=1)
    shuffled_out_degree = np.sum(shuffled_w != 0, axis=1)
    assert np.array_equal(original_out_degree, shuffled_out_degree), "Out-degree sequence must remain unchanged."
    
    original_in_degree = np.sum(original_w != 0, axis=0)
    shuffled_in_degree = np.sum(shuffled_w != 0, axis=0)
    assert np.array_equal(original_in_degree, shuffled_in_degree), "In-degree sequence must remain unchanged."

    # Check the total weights are unchanged
    original_total_weight = np.sum(original_w)
    shuffled_total_weight = np.sum(shuffled_w)
    assert np.isclose(original_total_weight, shuffled_total_weight), "Total weight of edges must remain unchanged."
    
    print("Test passed successfully!")

def perform_trade(w, index1, index2):
    """Perform trade by reshuffling neighbors while preserving weight distribution."""
    n = w.shape[0]

    # Locate out-neighbors (non-zero entries) for both nodes
    out_neighbors1 = w[index1, :] != 0
    out_neighbors2 = w[index2, :] != 0

    # Find unique neighbors for each node
    unique_to_1 = np.where(out_neighbors1 & ~(out_neighbors2 | (np.arange(n) == index2)))[0]
    unique_to_2 = np.where(out_neighbors2 & ~(out_neighbors1 | (np.arange(n) == index1)))[0]

    # Form the union of the unique sets
    union_unique = np.concatenate((unique_to_1, unique_to_2))

    # Remove the unique elements from their original nodes
    original_weights1 = w[index1, unique_to_1]
    original_weights2 = w[index2, unique_to_2]
    w[index1, unique_to_1] = 0
    w[index2, unique_to_2] = 0

    # Shuffle for randomness in selection
    np.random.shuffle(union_unique)
    
    # Select new neighbors for index1 and index2
    selected_for_1 = union_unique[:len(unique_to_1)]
    selected_for_2 = union_unique[len(unique_to_1):]

    # Redistribute weights preserved within selections
    if len(selected_for_1) > 0:
        w[index1, selected_for_1] = original_weights1 * (np.sum(original_weights1) / np.sum(original_weights1))
    if len(selected_for_2) > 0:
        w[index2, selected_for_2] = original_weights2 * (np.sum(original_weights2) / np.sum(original_weights2))


def global_curveball_directed(w, n_iterations=1000):
    '''
    Perform Global Curveball on a directed, weighted graph

    Parameters
    ----------
    w : TYPE
        DESCRIPTION.
    n_iterations : TYPE, optional
        DESCRIPTION. The default is 1000.

    Returns
    -------
    w : TYPE
        DESCRIPTION.

    '''
    
    validate_adjacency_matrix(w)
    original_w = w.copy()
    n = w.shape[0]
    for _ in range(n_iterations):
        node_indices = np.arange(n)
        np.random.shuffle(node_indices)

        for i in range(0, n - 1, 2):
            perform_trade(w, node_indices[i], node_indices[i + 1])
    
    test_curveball_correctness(original_w, w)
    return w
