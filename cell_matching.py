# cell shuffle
from lap import lapjv
import numpy as np
import pandas as pd
import os
import torch
from geomloss import SamplesLoss
from scipy.optimize import linear_sum_assignment


blur = 0.05
scaling = 0.3
# latent_size = 100
p = 2
match_method = "Hungarian"
# match_method = "OT"
input_path = '/gpfs/data/rsingh47/ylei29/CS2952G/schiebinger2019/large_test/original'
output_path = '/gpfs/data/rsingh47/ylei29/CS2952G/schiebinger2019/large_test/hungarian-original'
file_names = [f for f in os.listdir(input_path) if (f.endswith('.tsv.gz'))]

def generate_cost(X1, X2):
    n, m = X1.shape
    cost_matrix = np.zeros((n, n))
    print(f"N, M: {n}, {m}")
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = np.sum((X1[i, :] - X2[j, :]) ** 2)

    return cost_matrix

def Hungarian(cost_matrix):
    # Apply the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Calculate the total cost
    total_cost = cost_matrix[row_ind, col_ind].sum()
    # Create a mapping from set 1 to set 2
    mapping = list(zip(row_ind, col_ind))
    return mapping, total_cost

def shuffle_matrix(X1, mapping):
    shuffled_X1 = np.zeros_like(X1)
    for i, j in mapping:
        shuffled_X1[j, :] = X1[i, :]
    return shuffled_X1

def OT(rc, rc_p):
    OT_solver = SamplesLoss(loss = "sinkhorn", p = 2, blur = blur, scaling=scaling, 
                            debias = False, potentials = True)    
    N, M, D = rc.shape[0], rc_p.shape[0], rc.shape[1]  # Number of points, dimension
    t_weight = torch.ones(N)/N
    tp1_weight = torch.ones(M)/M
    print('Computing Wasserstein')
    F_pot, G_pot = OT_solver(t_weight, rc, tp1_weight, rc_p)  # Dual potentials
    
    a_i, x_i = t_weight.view(N,1), rc.view(N,1,D)
    b_j, y_j = tp1_weight.view(1,M), rc_p.view(1,M,D)
    F_pot_i, G_pot_j = F_pot.view(N,1), G_pot.view(1,M)

    C_ij = (1/p) * ((x_i - y_j)**p).sum(-1)  # (N,M) cost matrix
    eps = blur**p  # temperature epsilon
    P_ij = ((F_pot_i + G_pot_j - C_ij) / eps).exp() * (a_i * b_j)  # (N,M) transport plan'
    coupling_mat = P_ij.detach().numpy()

    # use lapjv to find the best alignment
    _, _, matching = lapjv(coupling_mat)
    return matching


# List to store numpy arrays
mat = []

# Read each file and store it as a numpy array in the list
for file in file_names:
    file_path = os.path.join(input_path, file)
    df = pd.read_csv(file_path, sep='\t', compression='gzip')
    array = df.values
    mat.append(array[:,1:])

print(mat[0].shape)
# exit()
# for loop

df = pd.DataFrame(mat[0])
output_filename = f'{output_path}/1.tsv.gz'
df.to_csv(output_filename, sep='\t', index=False, compression = 'gzip')

for i in range(len(mat)-1):
    print(i)
    rc = np.array(mat[i],dtype=float)
    rc_p = np.array(mat[i+1],dtype=float)
    # calculate ot loss, get the coupling matrix
    if match_method == 'OT':
        rc = torch.tensor(rc)
        rc_p = torch.tensor(rc_p)
        maching = OT(rc, rc_p)
        print(f"Now time is {i} and matching is \n {matching}")
        # shuffle the last matrix
        mat[i+1] = rc_p[matching]

    elif match_method == 'Hungarian':
        print("Computing Hungarian")
        cost = generate_cost(rc_p, rc)
        mapping, val = Hungarian(cost)
        mat[i+1] = shuffle_matrix(rc_p, mapping)

    df = pd.DataFrame(mat[i+1])
    output_filename = f'{output_path}/{i+2}.tsv.gz'
    df.to_csv(output_filename, sep='\t', index=False, compression = 'gzip')

# save the mats

# for i in range(len(mat)):
#     # Extract the (c,g) matrix for the i-th t
#     matrix = mat[i]
    
#     # Convert the numpy array to a pandas DataFrame
#     df = pd.DataFrame(matrix)
    
#     # Construct the file name
#     file_name = f'schiebinger2019/raw_data_origin/matched_raw_data/matched_original_{i+1}.tsv.gz'
    
#     # Save the DataFrame to a compressed TSV file
#     df.to_csv(file_name, sep='\t', index=False, compression='gzip')
