import numpy as np
import pandas as pd
# import 

mat1 = pd.read_csv('./test_large_latent_batch_1.tsv.gz', sep = '\t').values[:,1:]
mat2 = pd.read_csv('./test_large_latent_batch_2.tsv.gz', sep = '\t').values[:,1:]

def compare(mat1, mat2):
    mat = np.array(mat1 - mat2)
    print(mat.shape)
    mat = np.mean(mat, axis = 1)
    # print(mat[:20])
    mat = mat ** 2 
    return mat.sum()

m = compare(mat1, mat2)
cnt=0
for i in range(100):
    np.random.shuffle(mat2)
    n = compare(mat1, mat2)
    if( n <= m):
        cnt+=1

print(cnt)

# print(mat.sum())