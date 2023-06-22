from cProfile import label
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from pdb import set_trace as st


# Path to the labels
file_path = r'../Motognosis-FEF/data/segmented_data_frames/'

# List label files
label_files = [e for e in os.listdir(file_path) if 'Labels.npy' in e]
nb_subjects = len(label_files)
print('%d subject(s) found' % nb_subjects)

# Sort label files per subject ID
subject_ids = []
for idx in range(nb_subjects):
    current_file = label_files[idx]
    pos = current_file.find(' - Labels.npy')
    subject_ids += [int(current_file[pos-3:pos])]
permutation = np.argsort(subject_ids)
label_files = list(np.array(label_files)[permutation])

# Get information to plot
subject_id = []
nb_fm = []
nb_no_fm = []
invalid = []
colors = {-1: 'red', 0: 'limegreen', 1: 'cyan'}

### Simple distribution plot
# for idx in range(nb_subjects):
#     # Extract subject ID
#     pos = label_files[idx].find(' - Labels.npy')
#     subject_id += [int(label_files[idx][pos-3:pos])]
#     # Get number of FM and not FM
#     tmp_labels = np.load(os.path.join(file_path,label_files[idx]))
#     unique, counts = np.unique(tmp_labels, return_counts=True)
#     occurences = dict(zip(unique, counts))
#     nb_fm += [occurences[1]]
#     nb_no_fm += [occurences[0]]
#     invalid += [occurences[-1]]

# # Plot bar graph
# b1 = plt.barh(subject_id,nb_fm,color='cyan')
# b2 = plt.barh(subject_id,nb_no_fm,color='green')
# b3 = plt.barh(subject_id,invalid,color='red')
# plt.legend([b1,b2,b3], ["FM", "No FM","invalid"],loc="upper right")
# plt.title('Repartition of FM and no FM per subject')
# plt.yticks(subject_id)
# plt.ylabel('Subject ID')
# plt.xlabel('Number of 1-second frames')
# plt.show()

# for idx in range(nb_subjects):
#     # Extract subject ID
#     pos = label_files[idx].find(' - Labels.npy')
#     subject_id += [int(label_files[idx][pos-3:pos])]
#     # Get number of FM and not FM
#     tmp_labels = np.load(os.path.join(file_path,label_files[idx]))
#     unique, counts = np.unique(tmp_labels, return_counts=True)
#     occurences = dict(zip(unique, counts))
#     nb_fm += [occurences[1]]
#     nb_no_fm += [occurences[0]]
#     invalid += [occurences[-1]]

### Function plot
# Plot subgraph
# fig, ax = plt.subplots(nb_subjects,1)
# for idx in range(nb_subjects):
#     tmp_labels = np.load(os.path.join(file_path,label_files[idx]))
#     ax[idx].step(list(range(1,len(tmp_labels)+1)),tmp_labels)

# plt.show()


## Manual plot with plt.barh()
for idx in range(nb_subjects):
    # Extract subject ID
    pos = label_files[idx].find(' - Labels.npy')
    subject_str = label_files[idx][pos-3:pos]
    subject_idx = int(subject_str)
    # Load the labels
    tmp_labels = np.load(os.path.join(file_path,label_files[idx]))
    nb_windows = len(tmp_labels)
    # Plot the label repartition
    current_idx = 0
    while current_idx < nb_windows-1:
        #print('Current idx = %d' % current_idx)
        # Get current label
        current_label = tmp_labels[current_idx]
        #print('Current label = %d' % current_label)
        # Get all labels until they change
        increment_idx = 0
        while current_idx+increment_idx < nb_windows-1 and tmp_labels[current_idx] == tmp_labels[current_idx+increment_idx]:
            increment_idx += 1
        # Plot the bar
        #plt.barh(idx+1,increment_idx,left=current_idx,color=colors[current_label])
        plt.barh(subject_idx,increment_idx,left=current_idx,color=colors[current_label])
        # Change the current idx
        current_idx += increment_idx
    # Write the subject idx
    #plt.text(idx+1,500,subject_str,fontsize=20,bbox=dict(facecolor='red', alpha=0.5))

# Plot bar graph 
plt.title('Repartition of FM/no FM/invalid data per subject')
#plt.legend(["no FM", "invalid","FM"],loc="upper right")
plt.grid(True)
red_patch = mpatches.Patch(color='r', label='invalid')
blue_patch = mpatches.Patch(color='cyan', label='FM')
green_patch = mpatches.Patch(color='limegreen', label='no FM')
plt.legend(handles=[red_patch,blue_patch,green_patch])
plt.ylabel('Subject ID')
plt.yticks(ticks=list(range(1,nb_subjects+1)),labels=[str(e) for e in subject_id]) # NOTE: y ticks do not work for some reason
ax = plt.gca()
ax.set_axisbelow(True)
# for bar in ax.containers:
#     st()
#     ax.bar_label(bar)
# ax.set_yticks(list(range(1,nb_subjects+1)),labels=[str(e) for e in subject_id])
# ax.set_yticklabels([str(e) for e in subject_id])
plt.xlabel('Time (in seconds)')
plt.show()