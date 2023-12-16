import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
# import 
GENERATE_COST = False

def generate_cost(X1, X2):
    n, m = X1.shape
    cost_matrix = np.zeros((n, n))

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

def compare(mat1, mat2):
    mat = np.array(mat1 - mat2)
    # print(mat.shape)
    mat = np.mean(mat, axis = 1)
    # print(mat[:20])
    mat = mat ** 2 
    return mat.sum()

def shuffle_matrix(X1, mapping):
    shuffled_X1 = np.zeros_like(X1)
    for i, j in mapping:
        shuffled_X1[j, :] = X1[i, :]
    return shuffled_X1
    

mat1 = pd.read_csv('./1.tsv.gz', sep = '\t').values
mat2 = pd.read_csv('./2.tsv.gz', sep = '\t').values
n = mat1.shape[0]

# print(mapping[:])
ot = compare(mat1, mat2)
print(f"Optimal Transport loss: {ot}")

if GENERATE_COST:
    cost = generate_cost(mat1, mat2)
else:
    cost = pd.read_csv('./cost_matrix.csv', sep = ',').values
    cost = cost[:,1:]

df= pd.DataFrame(cost)
df.to_csv("Cost", sep=',')
# cost = np.random.rand(n,n)
# print(cost.shape)
# print(cost[:5])
mapping, total_cost = Hungarian(cost)
mat1_shuffled = shuffle_matrix(mat1, mapping)
hung = compare(mat1_shuffled, mat2)
print(f"Hung loss: {hung}")

cnt_ot=0
cnt_hung=0
trail = []

for i in range(100):
    np.random.shuffle(mat2)
    cur = compare(mat1, mat2)
    # print(f"Cur is {cur}")
    if cur < ot:
        cnt_ot+=1
    if cur < hung:
        cnt_hung+=1
    trail.append(cur)

trail_mean = np.mean(trail)
trail_var = np.sqrt(np.var(trail))
print(f"Trails has the distribution: {trail_mean}, {trail_var}")


# plt.scatter(range(100), n)
# plt.show()
# plt.savefig("pig.png")


print(f"OT is worse than: {cnt_ot}")
print(f"Hungerian is worse than: {cnt_hung}")

# print(mat.sum())