import pandas as pd
import torch
import os

df = pd.read_csv('./original_1.tsv.gz',sep='\t')
df.to_csv('./original_1.csv',sep=',')
