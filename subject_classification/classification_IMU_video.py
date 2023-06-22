import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from skopt.space import Integer, Real, Categorical
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datetime import datetime
from math import ceil
from pdb import set_trace as st
import csv
import re
from time import time


#################################################################################################################################################
### Meta parameters
data_folder = './data/hand_crafted_features' # Path where the features are saved in .npy format
label_folder = './data/segmented_data_frames' # Path where the features are saved in .npy format
result_path = './' # Path where to save csv file of results
imu_rgbd_synchronisation_path = '../test scripts/offsets_df.xlsx' # Path to the file containing the offsets
estimations_save_path = '../test scripts/rf_estimations/' # Path where to save the estimations of the classifier
classification = 'rf' # 'rf' or 'mlp' | NOTE: MLP not compatible with RFE
#nb_trees = 100 # Initial number of decision trees in the random forest for RFE
#max_depth = 10 # Initial maximum depth for the random forest for RFE
# Parameter for the Bayesian/grid search
search_params = {}
if classification == 'rf':
    search_params['n_estimators'] = (50,200) # Parameters to explore for number of trees
    search_params['max_depth'] = (15,50) # Parameters to explore for max depth
    search_params['max_features'] = ['sqrt','log2',None] # Parameters to explore for the mex number of features to consider when looking for the best split
    search_params['class_weight'] = [None] # Parameters to explore for class weights
elif classification == 'mlp':
    search_params['hidden_layer_sizes'] = [(100),(500,100,10),(50,50,50,50,50)]
    search_params['activation'] = ['relu']
    search_params['alpha'] = [1e-4,1e-6] # Coefficient of the L2 regulariser
    search_params['learning_rate'] = ['invscaling']
    
subjects_to_consider = ['032a','033','035','040','041','042']
features_to_consider = ['imu', 'mccay', 'marchi', 'marchi_prep', 'chambers', 'chambers_prep'] # List of elements in {'imu', 'mccay', 'marchi', 'marchi_prep', 'chambers', 'chamber_prep'}
rfe = True # Enable or disable RFE
step = 1 # RFE step size
save_ranking = '../test scripts/rfe_rankings/' # If not empty, save the ranking of features obtained by RFE for each subject
nb_features_to_select = 20 # Number or proportion of features to select with RFE
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
    
    # Perform Leave-One-Subject-Out cross validation
    subject_counter = 0

    for subject_index in subject_idx_table.keys():

        print('Preparing the data for subject %d ...' % subject_index)
        start = time()
        # Get associated data and label files for the test subject
        test_labels = subject_data[subject_index][1]
        test_data = subject_data[subject_index][0]

        # Build training set
        # nb_train_examples = 0 # Compute the size of the training data for faster numpy initialisation
        train_subjects = [e for e in subject_idx_table.keys()]
        train_subjects.remove(subject_index)
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

        # Randomly shuffle training labels and data in unisson
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

        # Apply RFE
        if rfe:
            if classification == 'rf':
                start = time()
                rfe_classifier = RandomForestClassifier()   
                print('Applying RFE for subject %d ...' % subject_index)
                selector = RFE(rfe_classifier,step=step,n_features_to_select=nb_features_to_select)
                selector = selector.fit(train_data,train_labels)
                if len(save_ranking) > 0:
                    np.save(os.path.join(save_ranking,classification+'_rfe_ranking_'+str(subject_index).zfill(3)+'.npy'),selector.ranking_)
                train_data = train_data[:,selector.support_]
                test_data = test_data[:,selector.support_]
                end = time()
                print('RFE performed in %.2f seconds' % (end-start))
            elif classification == 'mlp':
                print('RFE not implemented for MLP classifier -> skipping this part')

        # Bayesian hyper-parameter optimisation
        start = time()
        if classification == 'rf':
            print('Starting Bayesian hyper-parameter optimisation for subject %d ...' % subject_index)
            search = BayesSearchCV(estimator=RandomForestClassifier(),search_spaces=search_params)
        elif classification == 'mlp':
            print('Starting grid-search hyper-parameter optimisation for subject %d ...' % subject_index)
            #search = BayesSearchCV(estimator=MLPClassifier(),search_spaces=search_params) # NOTE: GridSearchCV does not work with tuple configurations (e.g. hidden_layer_sizes)
            search = GridSearchCV(estimator=MLPClassifier(),param_grid=search_params)
        search.fit(train_data,train_labels)
        best_params = search.best_params_
        print(search.best_score_)
        print(best_params)
        end = time()
        print('Bayesian search performed in %.2f seconds' % (end-start))

        # Train classifier with optimal parameters returned by Bayesian search
        start = time()
        print('Training classifier for subject %d ...' % subject_index)
        if classification == 'rf':
            classifier = RandomForestClassifier(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                class_weight=best_params['class_weight'],
                max_features=best_params['max_features'])
        elif classification == 'mlp':
            classifier = MLPClassifier(
                hidden_layer_sizes=best_params['hidden_layer_sizes'],
                activation=best_params['activation'],
                alpha=best_params['alpha'],
                learning_rate=best_params['learning_rate'])

        classifier.fit(train_data,train_labels)
        end = time()
        print('Classifier trained in %.2f seconds' % (end-start))

        # Get classifier predictions and save them
        estimations = classifier.predict(test_data)
        np.save(os.path.join(estimations_save_path,classification+'_estimations_subject'+str(subject_index).zfill(3)+'.npy'),estimations)

        # Compute evaluation metrics
        current_accuracy = accuracy_score(test_labels,estimations)
        current_af1 = f1_score(test_labels,estimations,average='macro')
        accuracies[subject_counter] = 100*current_accuracy
        f1_scores[subject_counter] = 100*current_af1
        conf_mat = confusion_matrix(test_labels, estimations)
        print('    Accuracy = %.2f %%' % (100*current_accuracy))
        print('    AF1 = %.2f %%' % (100*current_af1))
        print(conf_mat)

        subject_counter += 1

    # Print results
    print('')
    print('######################################################################')
    print('Average accuracy: %.2f +- %.2f %%' % (np.mean(accuracies),np.std(accuracies)))
    print('Average AF1: %.2f +- %.2f %%' % (np.mean(f1_scores),np.std(f1_scores)))
    print('######################################################################')
    print('')

    # Write results in csv file
    current_time = datetime.now()
    file_name = str(current_time.year)+str(current_time.month).zfill(2)+str(current_time.day).zfill(2)+'_'+str(current_time.hour).zfill(2)+\
            str(current_time.minute).zfill(2)+str(current_time.second).zfill(2)+' - IMU LOSOCV Results.csv'

    with open(result_path+file_name,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Subject ID','Accuracy','F1 score'])
        for idx in range(len(accuracies)):
            writer.writerow([str(subject_indices[idx]).zfill(3),accuracies[idx],f1_scores[idx]])
        writer.writerow(['Average',np.mean(accuracies),np.mean(f1_scores)])
        writer.writerow(['Std',np.std(accuracies),np.std(f1_scores)])


