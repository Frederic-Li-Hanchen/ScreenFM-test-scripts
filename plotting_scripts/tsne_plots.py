import numpy as np
import os
import re
from pdb import set_trace as st
import pandas as pd
from math import ceil
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

#################################################################################################################################################
### Meta parameters
data_folder = '../Motognosis-FEF/data/hand_crafted_features' # Path where the features are saved in .npy format
label_folder = '../Motognosis-FEF/data/segmented_data_frames' # Path where the features are saved in .npy format
imu_rgbd_synchronisation_path = './offsets_df.xlsx' # Path to the file containing the offsets   
subjects_to_consider = ['031', '032a','033','035', '036', '037', '040', '041', '042']
features_to_consider = ['imu', 'mccay', 'marchi', 'marchi_prep', 'chambers', 'chambers_prep'] # List of elements in {'imu', 'mccay', 'marchi', 'marchi_prep', 'chambers', 'chamber_prep'}
rfe_average_ranks = './rfe_rankings/top20/average_RFE_ranks.csv' # Path to the CSV file containing the average ranks
nb_features_to_select = 20
#################################################################################################################################################

# Create table of correspondencies for subject indices: (key, value) = (id_int, id_str)
subject_idx_table = {}
for index in subjects_to_consider:
    char_present = re.search('[a-zA-Z]', index)
    if char_present is None:
        subject_idx_table[int(index)] = index
    else:
        subject_idx_table[int(index[:char_present.start()])] = index

nb_subjects = len(subjects_to_consider)
if nb_subjects <2:
    print('Not enough subjects to consider for the cross-validation! Process aborted.')
    exit()
accuracies = np.zeros(nb_subjects,dtype=np.float32)
f1_scores = np.zeros(nb_subjects,dtype=np.float32)
subject_indices = np.zeros(nb_subjects,dtype=int)

# Load relevant labels
label_list = os.listdir(label_folder)
label_list = [e for e in label_list if 'Labels.npy' in e]
labels_to_keep = []
for label_name in label_list:
    pos = label_name.find(' - ')
    subject_index  = int(label_name[pos+3:pos+6]) # Subject index encoded in a 3-digit number
    if subject_index in subject_idx_table.keys():
        labels_to_keep += [label_name]

# Load all relevant features
file_list = os.listdir(data_folder)
features_to_load = {}

if 'imu' in features_to_consider:
    feature_file_list = [e for e in file_list if 'HandCraftedFeaturesIMU.npy' in e]
    feature_to_keep = []
    for feature_name in feature_file_list:
        pos = feature_name.find(' - ')
        subject_index  = int(feature_name[pos+3:pos+6]) # Subject index encoded in a 3-digit number
        if subject_index in subject_idx_table.keys():
            feature_to_keep += [feature_name]
    feature_to_keep.sort()
    features_to_load['imu'] = feature_to_keep

if 'mccay' in features_to_consider:
    feature_file_list = [e for e in file_list if 'mccay2019.npy' in e]
    feature_to_keep = []
    for feature_name in feature_file_list:
        pos = feature_name.find('_moto_')
        if feature_name[pos-1].isdigit():
            subject_index = int(feature_name[pos-3:pos]) # Subject index encoded in a 3-digit number
        else: # e.g. '032a'
            subject_index = int(feature_name[pos-4:pos-1])
        if subject_index in subject_idx_table.keys():
            feature_to_keep += [feature_name]
    feature_to_keep.sort()
    features_to_load['mccay'] = feature_to_keep

if 'marchi' in features_to_consider:
    feature_file_list = [e for e in file_list if 'marchi2019.npy' in e]
    feature_to_keep = []
    for feature_name in feature_file_list:
        pos = feature_name.find('_moto_')
        if feature_name[pos-1].isdigit():
            subject_index = int(feature_name[pos-3:pos]) # Subject index encoded in a 3-digit number
        else: # e.g. '032a'
            subject_index = int(feature_name[pos-4:pos-1])
        if subject_index in subject_idx_table.keys():
            feature_to_keep += [feature_name]
    feature_to_keep.sort()
    features_to_load['marchi'] = feature_to_keep

if 'marchi_prep' in features_to_consider:
    feature_file_list = [e for e in file_list if 'marchi2019_with_prepro.npy' in e]
    feature_to_keep = []
    for feature_name in feature_file_list:
        pos = feature_name.find('_moto_')
        if feature_name[pos-1].isdigit():
            subject_index = int(feature_name[pos-3:pos]) # Subject index encoded in a 3-digit number
        else: # e.g. '032a'
            subject_index = int(feature_name[pos-4:pos-1])
        if subject_index in subject_idx_table.keys():
            feature_to_keep += [feature_name]
    feature_to_keep.sort()
    features_to_load['marchi_prep'] = feature_to_keep

if 'chambers' in features_to_consider:
    feature_file_list = [e for e in file_list if 'chambers2020.npy' in e]
    feature_to_keep = []
    for feature_name in feature_file_list:
        pos = feature_name.find('_moto_')
        if feature_name[pos-1].isdigit():
            subject_index = int(feature_name[pos-3:pos]) # Subject index encoded in a 3-digit number
        else: # e.g. '032a'
            subject_index = int(feature_name[pos-4:pos-1])
        if subject_index in subject_idx_table.keys():
            feature_to_keep += [feature_name]
    feature_to_keep.sort()
    features_to_load['chambers'] = feature_to_keep

if 'chambers_prep' in features_to_consider:
    feature_file_list = [e for e in file_list if 'chambers2020_with_prepro.npy' in e]
    feature_to_keep = []
    for feature_name in feature_file_list:
        pos = feature_name.find('_moto_')
        if feature_name[pos-1].isdigit():
            subject_index = int(feature_name[pos-3:pos]) # Subject index encoded in a 3-digit number
        else: # e.g. '032a'
            subject_index = int(feature_name[pos-4:pos-1])
        if subject_index in subject_idx_table.keys():
            feature_to_keep += [feature_name]
    feature_to_keep.sort()
    features_to_load['chambers_prep'] = feature_to_keep

# Load IMU-Kinect synchronisation information
imu_kinect_sync = pd.read_excel(imu_rgbd_synchronisation_path)

# Load and synchronise the data and labels for all subjects
subject_data = {}

for subject_index in subject_idx_table.keys():

    print('Preparing data and labels for subject %d ...' % subject_index)

    # Get associated label file for the current subject
    label_list = [e for e in labels_to_keep if str(subject_index).zfill(3)+' - Labels.npy' in e]
    label_file_name = label_list[0]
    labels = np.load(os.path.join(label_folder,label_file_name)) 

    # Retrieve data
    data_dictionary = {}
    for key in features_to_load.keys():
        # Look for the correct file path to be loaded
        paths = features_to_load[key]
        #file_to_load = [e for e in paths if ' - '+str(subject_index).zfill(3)+' - ' in e or '_'+str(subject_index).zfill(3)+'_' in e]
        file_to_load = [e for e in paths if ' - '+str(subject_index).zfill(3)+' - ' in e or '_'+subject_idx_table[subject_index]+'_' in e]
        # ### DEBUG
        # t = np.load(os.path.join(data_folder,file_to_load[0]),allow_pickle=True)
        # if np.isnan(t).any():
        #     st()
        data_dictionary[key] = np.load(os.path.join(data_folder,file_to_load[0]),allow_pickle=True)

    # NOTE: IMU and RGBD feature arrays do not have the same size, and need to be synchronised
    # Look for the IMU and Kinect offset
    if len(features_to_consider) >=2 or 'imu' not in features_to_consider:
        b = imu_kinect_sync['Filename Kinect'].str.contains('_'+subject_idx_table[subject_index]+'.more')
        subject_info = imu_kinect_sync.loc[b]
        offset = subject_info['First synced timestamp Iphone'].iloc[0] # in ms
        frame_offset = subject_info['Offset Frames'].iloc[0]

        # Remove unaligned segments at the beginning of the recording
        nb_frames_to_discard_begin = ceil(offset/1000)
        kinect_features = [e for e in features_to_consider if 'imu' not in e]
        if frame_offset < 0: # Kinect starts later than iPhone+IMU
            if 'imu' in features_to_consider:
                data_dictionary['imu'] = data_dictionary['imu'][nb_frames_to_discard_begin:]
            labels = labels[nb_frames_to_discard_begin:]
        elif frame_offset > 0: # iPhone+IMU starts later than Kinect
            for key in kinect_features:
                data_dictionary[key] = data_dictionary[key][nb_frames_to_discard_begin:]
        # Remove unaligned segments at the end of the recording
        if len(labels) > len(data_dictionary[kinect_features[0]]): # More IMU segments remaining than Kinect
            nb_frames_to_discard_end = len(labels)-len(data_dictionary[kinect_features[0]])
            if 'imu' in features_to_consider:
                data_dictionary['imu'] = data_dictionary['imu'][:-nb_frames_to_discard_end]
            labels = labels[:-nb_frames_to_discard_end]
        elif len(labels) < len(data_dictionary[kinect_features[0]]): # More Kinect segments remaining than IMU
            nb_frames_to_discard_end = len(data_dictionary[kinect_features[0]])-len(labels)
            for key in kinect_features:
                data_dictionary[key] = data_dictionary[key][:-nb_frames_to_discard_end]

    array_list = []
    for feature in features_to_consider:
        array_list += [data_dictionary[feature]]
    data = np.concatenate(array_list,axis=1)
    subject_data[subject_index] = (data,labels)

# Keep only the features selected by RFE in the end
rfe_rankings = pd.read_csv(rfe_average_ranks)
features_to_keep = rfe_rankings.loc[:nb_features_to_select-1]
features_to_keep = features_to_keep['feature_id'].to_list()
features_to_keep.sort()

data_reduced_list = []
label_list = []
subject_labels_list = []
for subject_id in subject_data.keys():
    tmp_data, tmp_labels = subject_data[subject_id]
    # Create a vector of subject labels
    tmp_subject_label = subject_id*np.ones(len(tmp_labels),dtype=int)
    data_reduced_list += [tmp_data[:,features_to_keep]]
    label_list += [tmp_labels]
    subject_labels_list += [tmp_subject_label]

# Concatenate the data from all subjects and compute t-SNE
all_data = np.concatenate(data_reduced_list,axis=0)
all_labels = np.concatenate(label_list)
all_subject_labels = np.concatenate(subject_labels_list)

embedding = TSNE(n_components=2,perplexity=5,early_exaggeration=10,init='pca',learning_rate='auto')
transformed_data = embedding.fit_transform(all_data)

# Create color map and match each color to a subject
colors = plt.cm.rainbow(np.linspace(0,1,nb_subjects))
color_matching = {}
all_subject_id = list(set(all_subject_labels))
for idx in range(nb_subjects):
    color_matching[all_subject_id[idx]] = idx

# Change transparency parameter
colors[:,3] = 0.9
markers = {0: '.', 1:'x'}

# DEBUG: alternative hard-coded colors
#colors = ['red','green','blue','orange','limegreen','black','cyan','chocolate','slategrey']

# Plot the t-SNE projections of the features for each subject and distinguish per class
plt.grid(True)

# for idx in range(len(transformed_data)):
#     plt.scatter(transformed_data[idx,0],transformed_data[idx,1],marker=markers[all_labels[idx]],c=colors[color_matching[int(all_subject_labels[idx])],:])

unique_subject_id = list(set(all_subject_id))
unique_subject_id.sort()

for subject_idx in unique_subject_id:
    # Determine the corresponding color
    print('Processing the features for subject %d ...' % subject_idx)
    current_color = list(colors[color_matching[subject_idx]]) # (RGB-alpha list)
    #current_color = colors[color_matching[subject_idx]]
    # Determine the data corresponding to FM
    fm_id = [(e==1 and all_subject_labels[i]==subject_idx) for i,e in enumerate(all_labels)]
    data_to_plot = transformed_data[fm_id]
    # Plot the data corresponding to FM
    #plt.scatter(data_to_plot[:,0],data_to_plot[:,1],marker='^',c=current_color,alpha=0.5)
    plt.scatter(data_to_plot[:,0],data_to_plot[:,1],marker=markers[1],c=current_color)
    # Determine the data corresponding to no FM
    no_fm_id = [(e==0 and all_subject_labels[i]==subject_idx) for i,e in enumerate(all_labels)]
    data_to_plot = transformed_data[no_fm_id]
    # Plot the data corresponding to FM
    plt.scatter(data_to_plot[:,0],data_to_plot[:,1],marker=markers[0],c=current_color)

# Create the plot legend
color_legend = [mpatches.Patch(color=colors[color_matching[idx]]) for idx in unique_subject_id]
color_label = [str(e) for e in unique_subject_id]
dot = mlines.Line2D([], [], color='k', marker='.', linestyle='None', markersize=5)
cross = mlines.Line2D([], [], color='k', marker='x', linestyle='None', markersize=5)
marker_legend = [dot, cross]
marker_label = ['no FM', 'FM']
plt.legend(color_legend+marker_legend,color_label+marker_label)

plt.show()

