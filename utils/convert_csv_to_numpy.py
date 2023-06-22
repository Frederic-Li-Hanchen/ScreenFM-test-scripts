import numpy as np
from pdb import set_trace as st
import os


folder_path = '../Motognosis-FEF/data/hand_crafted_features/'
file_list = os.listdir(folder_path)
file_list = [e for e in file_list if '_chambers2020.csv' in e]
nb_files = len(file_list)

for idx in range(nb_files):
    print('Processing file %s ...' % (file_list[idx]))
    data = np.genfromtxt(os.path.join(folder_path,file_list[idx]), delimiter=',', skip_header=True)
    # if np.isnan(data).any():
    #     st()
    np.save(os.path.join(folder_path,file_list[idx].replace('.csv','.npy')),data)