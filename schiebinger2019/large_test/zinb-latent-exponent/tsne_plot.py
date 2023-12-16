# read the tsv file and plot the tSNE
# python tsne_plot.py --input_file=../data/latent.tsv --output_file=../data/latent.png

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='original_10.tsv')
parser.add_argument('--output_file', type=str, default='original_10.png')
args = parser.parse_args()

# read the tsv file, remove the first row and first column
df = pd.read_csv(args.input_file, sep='\t', header=None)
df = df.iloc[1:, 1:]
df = df.astype('float32')

# tSNE
tsne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
Y = tsne.fit_transform(df)
plt.scatter(Y[:, 0], Y[:, 1])
plt.savefig(args.output_file)
