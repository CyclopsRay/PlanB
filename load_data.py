import numpy as np
import pandas as pd
import os
import random
import scipy
import scipy.io
import torch
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix
from scipy.stats import ks_2samp, pearsonr, spearmanr  #cramervonmises_2samp, entropy
from scipy.special import softmax as scp_softmax
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances

def load_data(base_dir, study, truncate_time_point = 12, gene_num_cap = 4000, cell_num_cap = 4000):
    time_pt_list = np.arange(0, 8, 0.5)
    num_time_pts = len(time_pt_list)
    dense_mat_list = [] # This is a list, each element is a mat. Each mat could have totally different r and c.
    for t in np.arange(len(time_pt_list)):
        time_pt = time_pt_list[t]        
        str_time_pt = str(time_pt).replace('.', '_')
        fname = os.path.join(base_dir, study, 'gene_exp_mat_time_' + \
                            str_time_pt + '_100k_sf_1e04_rc.mtx')
    
        sparse_mat = scipy.io.mmread(fname) #coo format, only some couples of points. (rows, cols, value)
        # sparse_mat = coo_matrix.transpose(sparse_mat) #do not transpose for schiebinger
        sparse_mat = coo_matrix.tocsr(sparse_mat) #easier to index
        dense_mat = np.array(csr_matrix.todense(sparse_mat))
        dense_mat_list.append(dense_mat)
    
    print('Truncate time point (included in structure training):', truncate_time_point)
    print('Last time point:', len(dense_mat_list) - 1)
    print('Max number of time points (length):', len(dense_mat_list), '\n')
    print(f'------First element of the dense mat: {len(dense_mat_list[0])}, {len(dense_mat_list[0][0])}\n')
    
    num_cells_per_tp_list_pre = [np.size(dense_mat_list[i], 0) for i in np.arange(len(dense_mat_list))]
    min_num_cells = min(num_cells_per_tp_list_pre)
    print('Minimum number of cells over all time points:', min_num_cells, '\n')
        
    num_tps_total = len(dense_mat_list)

    # Before they have different size. Now unify them, and make them like (cell, time, gene) for further use.

    print("Number sample")
    new_dense_mat_list = []
    old_dense_mat_list = dense_mat_list
    for t in np.arange(len(time_pt_list)):
        dense_mat = dense_mat_list[t]
        cells_per_tp = np.shape(dense_mat)[0]
        
        
        if cells_per_tp < cell_num_cap:
            idcs = np.random.choice(cells_per_tp, size=min_num_cells, replace=True) # Add some points i.i.d.
        else:
            idcs = np.random.choice(cells_per_tp, size=min_num_cells, replace=False)
        dense_mat = dense_mat[idcs, :]
        new_dense_mat_list.append(dense_mat)
    dense_mat_list = new_dense_mat_list       
    for t in range(truncate_time_point+1):
            print(dense_mat_list[t].shape)

    # One hot embedding:-----------]

    one_hot_mat_all_tps_list = []
    
    for t in torch.arange(len(dense_mat_list)):
        categorical_tp = t*torch.ones((cell_num_cap))
        categorical_tp = categorical_tp.type(torch.long)
        one_hot_tp = torch.nn.functional.one_hot(categorical_tp, num_classes= len(dense_mat_list))
        one_hot_mat_all_tps_list.append(one_hot_tp.numpy())
    
    one_hot_mat_all_tps = np.concatenate(one_hot_mat_all_tps_list, axis=0)

    return dense_mat_list, one_hot_mat_all_tps
    
    # Threshold??
    