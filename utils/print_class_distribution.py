import numpy as np
from pdb import set_trace as st
import os

subject_id = '042'
path_to_labels = '../Motognosis-FEF/data/segmented_data_frames/'
label_list = os.listdir(path_to_labels)
label_file = [e for e in label_list if subject_id + ' - Labels.npy' in e]
path_to_label_file = os.path.join('../Motognosis-FEF/data/segmented_data_frames/',label_file[0])
labels = np.load(path_to_label_file)

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


