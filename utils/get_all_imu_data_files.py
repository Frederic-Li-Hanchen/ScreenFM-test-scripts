import os
from pdb import set_trace as st
from time import time
import shutil

from numpy import record


source_path = r'G:\raw\\'
target_path = r'C:\Users\Frederic_Li_Hanchen\Documents\Luebeck\ScreenFM\IMU data\\'

# List all subjects in numerical order
subject_list = os.listdir(source_path)
subject_list.sort()

# Loop on the subjects 
#for subject in subject_list:
for subject in subject_list[67:]:
    start = time()
    print('Copying data for subject %s ...' % subject)
    # Look for the relevant iPhone data
    path_to_iphone_folder = os.path.join(source_path, subject, 'Recordings')
    recording_list = os.listdir(path_to_iphone_folder)
    # Remove elements that are not folders in the list
    recording_list = [e for e in recording_list if os.path.isdir(os.path.join(path_to_iphone_folder, e))]
    recording_list.sort()

    # Only the last recording should be kept in normal cases (i.e. only 2 recordings)
    if len(recording_list)>2:
        print('    Warning: more than 2 iPhone recordings found for subject %s! Please double check for possible omissions.'%subject)
    target_folder = os.path.join(path_to_iphone_folder, recording_list[-1])

    # Get the two csv files
    folder_files = [e for e in os.listdir(target_folder) if '.csv' in e]
    # Copy them to the destination folder
    for file in folder_files:
        shutil.copy(os.path.join(target_folder,file),target_path)
    
    end = time()
    print('    Operation completed in %.2f seconds' % (end-start))

