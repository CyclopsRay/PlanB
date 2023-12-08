# cell shuffle
from lap import lapjv
import numpy as np
import pandas as pd
import os
import torch
from geomloss import SamplesLoss

blur = 0.05
scaling = 0.3
latent_size = 1478
p = 2


# load data
folder_path = './schiebinger2019/raw_data_origin/raw_data'
# output_path = 
file_names = [f for f in os.listdir(folder_path) if (f.startswith('original') and f.endswith('.tsv.gz'))]

# List to store numpy arrays
mat = []

# Read each file and store it as a numpy array in the list
for file in file_names:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path, sep='\t', compression='gzip')
    array = df.values
    mat.append(array[:,1:])

print(mat[0].shape)
# exit()
# for loop
for i in range(len(mat)-1):
    print(i)
    rc = np.array(mat[i],dtype=float)
    rc = torch.tensor(rc)
    rc_p = np.array(mat[i+1],dtype=float)
    rc_p = torch.tensor(rc_p)
    # calculate ot loss, get the coupling matrix
    OT_solver = SamplesLoss(loss = "sinkhorn", p = 2, blur = blur, scaling=scaling, 
                            debias = False, potentials = True)    
    N, M, D = rc.shape[0], rc_p.shape[0], latent_size  # Number of points, dimension
    t_weight = torch.ones(N)/N
    tp1_weight = torch.ones(M)/M
    # print('Computing Wasserstein')


    F_pot, G_pot = OT_solver(t_weight, rc, tp1_weight, rc_p)  # Dual potentials
    
    a_i, x_i = t_weight.view(N,1), rc.view(N,1,D)
    b_j, y_j = tp1_weight.view(1,M), rc_p.view(1,M,D)
    F_pot_i, G_pot_j = F_pot.view(N,1), G_pot.view(1,M)

    C_ij = (1/p) * ((x_i - y_j)**p).sum(-1)  # (N,M) cost matrix
    eps = blur**p  # temperature epsilon
    P_ij = ((F_pot_i + G_pot_j - C_ij) / eps).exp() * (a_i * b_j)  # (N,M) transport plan'
    coupling_mat = P_ij.detach().numpy()

    # use lapjv to find the best alignment

    _, matching, _ = lapjv(coupling_mat)
    print(f"Now time is {i} and matching is \n {matching}")
    # shuffle the last matrix
    mat[i+1] = rc_p[matching]

# save the mats

for i in range(len(mat)):
    # Extract the (c,g) matrix for the i-th t
    matrix = mat[i]
    
    # Convert the numpy array to a pandas DataFrame
    df = pd.DataFrame(matrix)
    
    # Construct the file name
    file_name = f'schiebinger2019/raw_data_origin/matched_raw_data/matched_original_{i+1}.tsv.gz'
    
    # Save the DataFrame to a compressed TSV file
    df.to_csv(file_name, sep='\t', index=False, compression='gzip')
