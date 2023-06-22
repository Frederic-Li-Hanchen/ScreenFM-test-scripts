import os
from pdb import set_trace as st
from time import time
import shutil

from numpy import record


source_path = r'G:\raw\\'
target_path = r'G:\to-be-preprocessed2\\'

# List all subjects in numerical order
subject_list = os.listdir(source_path)
subject_list.sort()

# Loop on the subjects 
#for subject in subject_list:
for subject in subject_list[67:]: # TODO: change starting subject number accordingly
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
    folder_to_copy = os.path.join(path_to_iphone_folder, recording_list[-1])
    # Create the target folder
    destination = os.path.join(target_path, subject)
    if not os.path.isdir(destination):
        os.makedirs(destination)
    # Copy the source folder to target
    shutil.copytree(folder_to_copy,destination,dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.json','*.fbd'))
    end = time()
    print('    Operation completed in %.2f seconds' % (end-start))
