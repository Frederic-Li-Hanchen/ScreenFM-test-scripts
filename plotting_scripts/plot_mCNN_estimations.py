import os
import numpy as np
from pdb import set_trace as st
import matplotlib.pyplot as plt


### Load input data
data_path = '../Motognosis-FEFv2/test_data_source/segmented_data_frames/'
list_data_files = os.listdir(data_path)
list_data_files = [e for e in list_data_files if ' - Labels.npy' in e]
# Extract subject ID information
pos = list_data_files[0].find(' - Labels.npy')
list_id = [e[pos-3:pos] for e in list_data_files]
list_id = list(set(list_id))
nb_subjects = len(list_id)

### Save path
save_path = './mCNN_estimation_plots/'

### Loop on the input data
for idx in range(nb_subjects):
    label_file_name = [e for e in list_data_files if ' - Labels.npy' in e and '- '+list_id[idx]+' -' in e][0]
    label_file = np.load(os.path.join(data_path,label_file_name))
    # Plot the time series
    plt.plot(label_file)
    plt.ylabel('Estimation')
    plt.xlabel('Time (in seconds)')
    plt.savefig(os.path.join(save_path,'ground_truth_'+list_id[idx]+'.png'))
    plt.clf()