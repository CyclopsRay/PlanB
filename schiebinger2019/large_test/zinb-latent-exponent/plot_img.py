# plot the images, in the tsv file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read in the data
df = pd.read_csv("./zinb_wave_latent_k2.tsv", sep='\t', index_col=0)
# labels = [1,10,11,12,13,14,15,16,2,3,4,5,6,7,8,9]
# plot the images
plt.figure(figsize=(10, 10))
x = df.iloc[:, 0].values
y = df.iloc[:, 1].values
for i in range(16):
    plt.scatter(x[i*1948:(i+1)*1948], y[i*1948:(i+1)*1948], label=i, s=1)
# plt.scatter(x, y, s=1)
legend = plt.legend(ncol=4)
for handle in legend.legendHandles:
    handle.set_sizes([30])
plt.savefig("./zinb_wave_latent_k2_wo_batcheff.png")