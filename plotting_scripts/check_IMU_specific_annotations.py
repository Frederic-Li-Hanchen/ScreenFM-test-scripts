import numpy as np
import os
from pdb import set_trace as st
import matplotlib.pyplot as plt


# Path to the label files
label_folder = '../Motognosis-FEFv2/test_data_source/segmented_data_frames/'

# Filter files per subject index
subject_id = '025'
list_files = [e for e in os.listdir(label_folder) if ' - ' + subject_id + ' - ' in e and 'Labels' in e]
list_files.sort()

# Create subplot and plot the various labels
fig, ax = plt.subplots(5,1)
for idx in range(len(list_files)):
    ax[idx].plot(np.load(os.path.join(label_folder,list_files[idx])))
    ax[idx].set_title(list_files[idx])

plt.show()