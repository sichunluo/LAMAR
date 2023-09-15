import pandas as pd
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error,mean_absolute_error


df = pd.read_csv('result_llm.csv', header=None)
df.columns = ['u','v','r']
df = df.sort_values('u').reset_index(drop=True)
df2 = pd.read_csv('sample.txt',  sep='\t')
df2.columns = ['u','v','r','t']
idx_l = []

pred_fm = np.load('pred_ans_DeepFM.npy')

alpha1 = 0.1
alpha2 = 0.3

for index, row in df.iterrows():
    count = (df['u'] == row['u']).sum()
    if count>80:
        idx_l.append(alpha1)
    else:
        idx_l.append(alpha2)

true_v = np.load('true_ans.npy')
l = []
l2 = []
l3 = []
pred_fm = np.array([i[0] for i in pred_fm])
temp_v = df['r'].values
temp_v =[temp_v[i]*idx_l[i] + pred_fm[i]* (1-idx_l[i]) for i in range(len(temp_v))]
l = temp_v 

a = l
b = true_v
print("RMSE:", round(np.sqrt(mean_squared_error(a, b)),4), '\nMAE:',round(mean_absolute_error(a, b),4))