import numpy as np
import os
import glob
from pdb import set_trace as st
import csv
import pandas as pd

############################################# Hyper-parameters #############################################
# Folder containing the files
file_path = r'./labels/033/'
############################################################################################################

def check_containment(file_path1, file_path2):
    # Read csv files
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)

    # Filter frames to keep only 'Text' entries
    df1 = df1.loc[df1['type']=='Text']
    df2 = df2.loc[df2['type']=='Text']

    # NOTE: issues when working directly with pandas frames -> conversion to numpy array instead
    a1 = df1.to_numpy()
    a2 = df2.to_numpy()

    if len(a2)>len(a1):
        return np.isin(a1,a2).all()
    else:
        return np.isin(a2,a1).all()


if __name__=='__main__':

    # List all label files by increasing file size
    # file_list = filter(os.path.isfile, glob.glob(file_path + '*.csv'))
    # file_list = sorted(file_list, key=lambda x: os.stat(x).st_size)
    file_list = os.listdir(file_path)
    sorted(file_list)

    # Get the largest file
    largest_file_path = os.path.join(file_path+file_list[-1])

    # Check if all smaller files are contained in the largest one
    containment = True

    for file in file_list[:-1]:
        containment = containment and check_containment(os.path.join(file_path,file), largest_file_path)

    print('###################################################')
    print(containment)
    print('###################################################')