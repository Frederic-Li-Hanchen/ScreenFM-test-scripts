import numpy as np
from pdb import set_trace as st
import os
import pandas as pd

#################################################################################################################
### Script that saves all class distributions to two CSV files: one with invalid counted in, the other without
#################################################################################################################

# Locate the label files
path_to_labels = '../Motognosis-FEF/data/segmented_data_frames/'
label_list = [e for e in os.listdir(path_to_labels) if ' - Labels.npy' in e]

# Create panda frames
path_to_save_results = './'
frame_with_invalid = pd.DataFrame(columns=['subject ID','invalid #','invalid %','FM #','FM %','no FM #','no FM %'],index=None)
frame_without_invalid = pd.DataFrame(columns=['subject ID','FM #','FM %','no FM #','no FM %'],index=None)

# Loop on the subjects
for idx in range(len(label_list)):
    file_name = label_list[idx]
    path_to_label_file = os.path.join(path_to_labels,file_name)
    labels = np.load(path_to_label_file)

    # Extract subject ID
    pos = file_name.find(' - Labels.npy')
    subject_id = int(file_name[pos-3:pos])

    # Count invalid, FM and no FM segments
    nb_frames = len(labels)
    invalid_count = 0
    no_fm_count = 0
    fm_count = 0

    for l in labels:
        if l == -1:
            invalid_count += 1
        elif l == 0:
            no_fm_count += 1
        elif l == 1:
            fm_count += 1

    print('')
    print('##########################################################################')
    print('Analysing label distribution for file %s' % path_to_label_file)
    print('--------------------------------------------------------------------------')
    print('With invalid frames counted')
    print('--------------------------------------------------------------------------')
    print('Total number of frames: %d' % nb_frames)
    print('Number of invalid (-1): %d (%.2f%%)' % (invalid_count,100*invalid_count/nb_frames))
    print('Number of no FM (0): %d (%.2f%%)' % (no_fm_count,100*no_fm_count/nb_frames))
    print('Number of FM (1): %d (%.2f%%)' % (fm_count,100*fm_count/nb_frames))
    print('--------------------------------------------------------------------------')
    print('With invalid frames removed')
    print('--------------------------------------------------------------------------')
    print('Total number of frames: %d' % (nb_frames-invalid_count))
    print('Number of no FM (0): %d (%.2f%%)' % (no_fm_count,100*no_fm_count/(nb_frames-invalid_count)))
    print('Number of FM (1): %d (%.2f%%)' % (fm_count,100*fm_count/(nb_frames-invalid_count)))
    print('##########################################################################')
    print('')

    frame_with_invalid.loc[idx] = [subject_id,invalid_count,100*invalid_count/nb_frames,fm_count,100*fm_count/nb_frames,no_fm_count,100*no_fm_count/nb_frames]
    frame_without_invalid.loc[idx] = [subject_id,fm_count,100*fm_count/(nb_frames-invalid_count),no_fm_count,100*no_fm_count/(nb_frames-invalid_count)]

# Order frames by increasing subjects and save them
frame_with_invalid.sort_values('subject ID', inplace=True)
frame_without_invalid.sort_values('subject ID', inplace=True)
frame_with_invalid.to_csv(os.path.join(path_to_save_results,'class_repartition_with_invalid.csv'),sep=',',index=False)
frame_without_invalid.to_csv(os.path.join(path_to_save_results,'class_repartition_without_invalid.csv'),sep=',',index=False)