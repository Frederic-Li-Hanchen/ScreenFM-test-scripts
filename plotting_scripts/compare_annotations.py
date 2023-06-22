from cProfile import label
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pdb import set_trace as st


# Path to the two labels files to compare
file_path1 = r'../data/20230613_090934 - 080 - Labels Andrea Kock Friederike Pagel.npy'
file_path2 = r'../data/20230613_091132 - 080 - Labels Andrea Kock.npy'

# Load labels
labels1 = np.load(file_path1)
labels2 = np.load(file_path2)

# Get information to plot
colors = {-1: 'red', 0: 'limegreen', 1: 'cyan'}

### Manual plot with plt.barh() for rater 1
nb_windows = len(labels1)
current_idx = 0
while current_idx < nb_windows-1:
    #print('Current idx = %d' % current_idx)
    # Get current label
    current_label = labels1[current_idx]
    #print('Current label = %d' % current_label)
    # Get all labels until they change
    increment_idx = 0
    while current_idx+increment_idx < nb_windows-1 and labels1[current_idx] == labels1[current_idx+increment_idx]:
        increment_idx += 1
    # Plot the bar
    plt.barh(1,increment_idx,left=current_idx,color=colors[current_label])
    #plt.barh(subject_idx,increment_idx,left=current_idx,color=colors[current_label])
    # Change the current idx
    current_idx += increment_idx

### Manual plot with plt.barh() for rater 2
nb_windows = len(labels2)
current_idx = 0
while current_idx < nb_windows-1:
    #print('Current idx = %d' % current_idx)
    # Get current label
    current_label = labels2[current_idx]
    #print('Current label = %d' % current_label)
    # Get all labels until they change
    increment_idx = 0
    while current_idx+increment_idx < nb_windows-1 and labels2[current_idx] == labels2[current_idx+increment_idx]:
        increment_idx += 1
    # Plot the bar
    plt.barh(2,increment_idx,left=current_idx,color=colors[current_label])
    #plt.barh(subject_idx,increment_idx,left=current_idx,color=colors[current_label])
    # Change the current idx
    current_idx += increment_idx

plt.title('Repartition of FM/no FM/invalid data per subject')
red_patch = mpatches.Patch(color='red', label='invalid')
blue_patch = mpatches.Patch(color='cyan', label='FM')
green_patch = mpatches.Patch(color='limegreen', label='no FM')
plt.legend(handles=[red_patch,blue_patch,green_patch])
plt.ylabel('Rater ID')
plt.xlabel('Time (in seconds)')
plt.show()