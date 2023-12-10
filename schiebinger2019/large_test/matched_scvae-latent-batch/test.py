import pandas as pd
df = pd.read_csv('./1.tsv.gz', sep='\t')
print(df.values)