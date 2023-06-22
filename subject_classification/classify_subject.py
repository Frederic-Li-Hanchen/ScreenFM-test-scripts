import numpy as np
import os
from pdb import set_trace as st
import pandas as pd

#################################################################################################################
### Script that saves the infant-level classification results in a CSV file
### Assumes that classifier estimations (1D numpy array) are available for each subject
### Can perform classification using the whole record, or a limited time horizon (e.g. 60 seconds)
#################################################################################################################

#################################################################################################################
### Hyper-parameters
#################################################################################################################

# Classification model to consider
# NOTE: only used to load the correct vector of estimations and in the saved filename
classifier = 'rf' # 'rf' or 'mlp'

# Path where the classifier estimations are saved for each subject
# NOTE: currentyl assumed to be saved under the name classifier_name+'_estimations_subject'+subject_idx+'.npy'
estimation_path = './rf_estimations/top20/'
#estimation_path = './rf_estimations/'

# Path to save the results CSV table
result_path = './results/'

# Thresholds on the number of FM to test
thresholds = [10,20,30,40,50,60,70,80,90]
nb_thresholds = len(thresholds)

# Time segment length to provide input to the classifier (in seconds)
# NOTE: If set to negative value, disable segmentation to work on the whole record
time_segment_length = 60

# Label path to determine how many windows are mostly invalid
label_path = '../Motognosis-FEF/data/segmented_data_frames/'

# Subjects to consider and associated labels
# 0: normal | 1: at risk
subject_labels = {
    '031': 1,
    '032': 0,
    '033': 0,
    '034': 1,
    '035': 0,
    '036': 0,
    '037': 0,
    '038': 0,
    '039': 0,
    '040': 0,
    '041': 1,
    '042': 1,
    '043': 0,
    '044': 1,
    '045': 0,
    '046': 1,
    '047': 0,
    '048': 1,
    '049': 0,
}
nb_subjects = len(subject_labels.keys())


#################################################################################################################
### Main code
#################################################################################################################
if time_segment_length <= 0:
    # Get the proportion of FM detected for each subject
    fm_proportion = {}
    for subject_idx in subject_labels.keys():
        estimations = np.load(os.path.join(estimation_path,classifier+'_all_estimations_subject'+subject_idx+'.npy'))
        nb_estimations = len(estimations)
        fm_proportion[subject_idx] = 100*np.sum(estimations)/nb_estimations

    # Get table of infant-level classification results using a simple threshold rule on FMs
    classification_table = np.zeros((nb_thresholds,nb_subjects), dtype=int)
    for i in range(nb_thresholds):
        j = 0
        for subject_idx in subject_labels.keys():
            if fm_proportion[subject_idx] >= thresholds[i]:
                classification_table[i,j] = 0
            else:
                classification_table[i,j] = 1
            j+=1

    # Check if the classification estimations match the actual infant label
    result_table = np.zeros((nb_thresholds,nb_subjects), dtype=int)
    for i in range(nb_thresholds):
        j = 0
        for subject_idx in subject_labels.keys():
            result_table[i,j] = (subject_labels[subject_idx] == classification_table[i,j])
            j+=1        

    # Save the result table in a csv file
    df = pd.DataFrame(result_table,index=thresholds,columns=list(subject_labels.keys()))
    df.to_csv(os.path.join(result_path,classifier+'_infant_level_classification_whole_record.csv'))

else: # Segmentation required

    # Load the labels of each subject
    subject_segment_labels = {}
    label_list = [e for e in os.listdir(label_path) if ' Labels.npy' in e]
    for subject_idx in subject_labels.keys():
        current_label_file = [e for e in label_list if ' - '+subject_idx+' - ' in e]
        subject_segment_labels[subject_idx] = np.load(os.path.join(label_path,current_label_file[0]))

    # Compute the proportion of FMs for each window
    estimation_windows = {}
    nb_invalid_windows = {} # A window is counted as invalid if more than half of its segments are labeled as invalid
    fm_proportion = {}
    for subject_idx in subject_labels.keys():
        all_windows = []
        all_fm_proportions = []
        invalid_counter = 0
        estimations = np.load(os.path.join(estimation_path,classifier+'_all_estimations_subject'+subject_idx+'.npy'))
        current_subject_labels = subject_segment_labels[subject_idx]
        nb_estimations = len(estimations)
        if nb_estimations < time_segment_length:
            print('Warning: less estimations than the chosen window size!')
            all_windows += [estimations]
            all_fm_proportions += [100*np.sum(estimations)/nb_estimations]
            tmp_array = (current_subject_labels==-1)
            if np.sum(tmp_array) >= time_segment_length/2:
                invalid_counter += 1
        else:
            current_idx = 0
            while current_idx+time_segment_length<nb_estimations:
                tmp_window = estimations[current_idx:current_idx+time_segment_length]
                tmp_array = (current_subject_labels[current_idx:current_idx+time_segment_length]==-1)
                if np.sum(tmp_array) >= time_segment_length/2:
                    invalid_counter += 1
                all_windows += [tmp_window]
                current_idx += time_segment_length
                all_fm_proportions += [100*np.sum(tmp_window)/time_segment_length]
        estimation_windows[subject_idx] = all_windows
        fm_proportion[subject_idx] = all_fm_proportions
        nb_invalid_windows[subject_idx] = invalid_counter

    # Check if the classification estimations match the actual infant label
    result_table = pd.DataFrame(columns=['threshold']+[e for e in subject_labels.keys()]+['nb_correctly_classified', 'proportion_correctly_classified'])
    result_table['threshold'] = thresholds
    threshold_idx = 0
    for t in thresholds:
        total_nb_correctly_classified = 0
        total_nb_segments = 0
        for subject_idx in subject_labels.keys():
            nb_correct_classification = 0
            current_fm_proportions = fm_proportion[subject_idx]
            nb_invalid = nb_invalid_windows[subject_idx]
            nb_segments = len(current_fm_proportions)
            for segment_idx in range(nb_segments):
                class_label = int(not(current_fm_proportions[segment_idx] >= t))
                if class_label == subject_labels[subject_idx]:
                    nb_correct_classification += 1
            result_table[subject_idx].iloc[threshold_idx] = (nb_correct_classification,nb_segments,nb_invalid)
            total_nb_correctly_classified += nb_correct_classification
            total_nb_segments += nb_segments
        result_table['nb_correctly_classified'].iloc[threshold_idx] = (total_nb_correctly_classified,total_nb_segments)
        result_table['proportion_correctly_classified'].iloc[threshold_idx] = 100*total_nb_correctly_classified/total_nb_segments
        threshold_idx += 1

    # Save the result table in a csv file
    result_table.to_csv(os.path.join(result_path,classifier+'_infant_level_classification_window_length_'+str(time_segment_length)+'.csv'),index=False)
    

    
