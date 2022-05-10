import pandas as pd
import numpy as np
from tqdm import tqdm

data_row = pd.read_csv("./data/9606.protein.sequences.v11.0.csv")

df = pd.DataFrame(data_row)
data = []
data_seq = ''
data_name = ''
# print(len(df[df.keys()]))
for i in tqdm(range(len(df[df.keys()]))):
    if df[df.keys()].values[i][0][0] == '>':
        if data_name != '' and data_seq != '':
            data.append([data_name, data_seq])
        data_name = df[df.keys()].values[i][0]
        data_seq = ''
    else:
        data_seq += df[df.keys()].values[i][0]

data.to_csv('test')
