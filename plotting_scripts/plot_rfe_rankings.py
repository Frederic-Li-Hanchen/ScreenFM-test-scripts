import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datetime import datetime
from math import ceil
from pdb import set_trace as st
import csv
import re
from time import time

#################################################################################################################################################
# Script that plots the graph accuracy in function of RFE iteration for one subject
#################################################################################################################################################

### Meta parameters
data_folder = '../Motognosis-FEF/data/hand_crafted_features' # Path where the features are saved in .npy format
label_folder = '../Motognosis-FEF/data/segmented_data_frames' # Path where the features are saved in .npy format
rfe_rank_path = './rfe_rankings/rf_rfe_ranking_032.npy'
result_path = './results/' # Path where to save the result plots.
imu_rgbd_synchronisation_path = './offsets_df.xlsx'
nb_trees = 100 # Number of decision trees in the random forest
max_depth = 10 # Maximum depth for the decision tree
#mlp_size = (100,50,10) # Layer sizes for a MLP classifier
subjects_to_consider = ['032a','033','035','040','041','042']
features_to_consider = ['imu', 'mccay', 'marchi', 'marchi_prep', 'chambers', 'chambers_prep'] # List of elements in {'imu', 'mccay', 'marchi', 'marchi_prep', 'chambers', 'chamber_prep'}
#################################################################################################################################################

### Main
if __name__ == '__main__':

    assert len(features_to_consider) >= 1
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
    #accuracies = np.zeros(nb_subjects,dtype=np.float32)
    #f1_scores = np.zeros(nb_subjects,dtype=np.float32)
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
            file_to_load = [e for e in paths if ' - '+str(subject_index).zfill(3)+' - ' in e or '_'+subject_idx_table[subject_index]+'_' in e]
            data_dictionary[key] = np.load(os.path.join(data_folder,file_to_load[0]),allow_pickle=True)

        # NOTE: IMU and RGBD feature arrays do not have the same size, and need to be synchronised
        # Look for the IMU and Kinect offset
        # NOTE: Kinect recording not necessarily shorter than IMU/iPhone
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
    
    # Load the RFE results
    rfe_ranks = np.load(rfe_rank_path)
    nb_rfe_iterations = len(set(rfe_ranks))
    accuracies = np.zeros(nb_rfe_iterations,dtype=np.float32) 
    f1_scores = np.zeros(nb_rfe_iterations,dtype=np.float32)
    # Determine the subject to be used as test set
    pos = rfe_rank_path.find('.npy')
    test_subject_idx = int(rfe_rank_path[pos-3:pos])

    print('Preparing the data for subject %d ...' % test_subject_idx)
    start = time()
    # Get associated data and label files for the test subject
    test_labels = subject_data[test_subject_idx][1]
    test_data = subject_data[test_subject_idx][0]

    # Build training set
    # nb_train_examples = 0 # Compute the size of the training data for faster numpy initialisation
    train_subjects = [e for e in subject_idx_table.keys()]
    train_subjects.remove(test_subject_idx)
    train_data_list = []
    train_label_list = []

    for idx in train_subjects:
        train_data_list += [subject_data[idx][0]]
        train_label_list += [subject_data[idx][1]]

    train_data = np.concatenate(train_data_list,axis=0)
    train_labels = np.concatenate(train_label_list)
    
    # Remove invalid examples from the sets
    train_idx_to_keep = [e!=-1 for e in train_labels]
    test_idx_to_keep = [e!=-1 for e in test_labels]
    train_data = train_data[train_idx_to_keep]
    train_labels = train_labels[train_idx_to_keep]
    test_data = test_data[test_idx_to_keep]
    test_labels = test_labels[test_idx_to_keep]

    # Randomly shuffle labels and data in unisson
    random_permutation = np.random.permutation(len(train_labels))
    train_data = train_data[random_permutation]
    train_labels = train_labels[random_permutation]

    # Replace any NaN value by 0
    if np.isnan(train_data).any():
        np.nan_to_num(train_data,copy=False,nan=0)
    if np.isnan(test_data).any():
        np.nan_to_num(test_data,copy=False,nan=0)

    end = time()
    print('Data prepared in %.2f seconds' % (end-start))

    # # Compute the classification results without feature selection
    # start = time()
    # print('Training classifier with all features ...')
    # classifier = RandomForestClassifier(n_estimators=nb_trees,max_depth=max_depth)
    # classifier.fit(train_data,train_labels)
    # end = time()
    # print('Classifier trained in %.2f seconds' % (end-start))
    # estimations = classifier.predict(test_data)
    # current_accuracy = 100*accuracy_score(test_labels,estimations)
    # current_af1 = 100*f1_score(test_labels,estimations,average='macro')
    # accuracies[0] = current_accuracy
    # f1_scores[0] = current_af1
    # conf_mat = confusion_matrix(test_labels, estimations)
    # print('    Accuracy = %.2f %%' % (current_accuracy))
    # print('    AF1 = %.2f %%' % (current_af1))
    # print(conf_mat)

    # Recompute the RFE results using the feature support information
    for idx in range(nb_rfe_iterations,0,-1):
        start = time()
        print('Training classifier for RFE iteration %d/%d ...' % (nb_rfe_iterations-idx+1,nb_rfe_iterations))
        # Determine the support of features to consider
        support = [e<=idx for e in rfe_ranks]
        train_data_tmp = train_data[:,support]
        test_data_tmp = test_data[:,support]
        #classifier = RandomForestClassifier(n_estimators=nb_trees,max_depth=max_depth)
        classifier = RandomForestClassifier()
        #classifier = MLPClassifier(hidden_layer_sizes=mlp_size)
        classifier.fit(train_data_tmp,train_labels)
        end = time()
        print('Classifier trained in %.2f seconds' % (end-start))
        # Get classifier predictions
        estimations = classifier.predict(test_data_tmp)
        # Compute evaluation metrics
        current_accuracy = 100*accuracy_score(test_labels,estimations)
        current_af1 = 100*f1_score(test_labels,estimations,average='macro')
        accuracies[nb_rfe_iterations-idx] = current_accuracy
        f1_scores[nb_rfe_iterations-idx] = current_af1
        conf_mat = confusion_matrix(test_labels, estimations)
        print('    Accuracy = %.2f %%' % (current_accuracy))
        print('    AF1 = %.2f %%' % (current_af1))
        print(conf_mat)

    # Plot the results of RFE in a graph
    x = list(range(nb_rfe_iterations))
    plt.plot(x,accuracies,color='r',label='accuracy',alpha=0.6)
    plt.plot(x,f1_scores,color='b',label='af1',alpha=0.6)
    plt.xlabel('RFE iteration')
    plt.ylabel('Evaluation metric')
    plt.title('RFE results for subject %d' % test_subject_idx)
    plt.legend()
    # Save the plot
    plt.savefig(os.path.join(result_path,'rfe_results_subject'+str(test_subject_idx).zfill(3)+'.png'))
    plt.show()

    



