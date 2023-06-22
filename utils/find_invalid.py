# Script to find the subjects who provided data annotated as invalid

import os
import numpy as np
import pandas as pd
from pdb import set_trace as st


# Path to folder containing annotations
path_to_folder = r'D:\ScreenFM\Annotation Data'

# Path to the result text file
path_to_text = './invalid_count.txt'

# List the folders for each subject
folder_list = [e for e in os.listdir(path_to_folder)]
folder_list.sort()

# Open file to write results
f = open(path_to_text,'w')

# Loop on each subject folder to check if it contains invalid data
for idx in range(len(folder_list)):
    current_subject = folder_list[idx]
    file_list = os.listdir(os.path.join(path_to_folder,current_subject))
    file_list = [e for e in file_list if 'Comments.csv' in e]
    # if current_subject == '018':
    #     st()
    csv = pd.read_csv(os.path.join(path_to_folder,current_subject,file_list[-1]),encoding='latin1')
    count = csv['content'].value_counts()
    if 'invalid_start' in count.keys():
        print("Invalid periods for subject %s: %1.f" % (current_subject,(count['invalid_start']+count['invalid_stop'])))
        f.write("Invalid periods for subject %s: %1.f\n" % (current_subject,(count['invalid_start']+count['invalid_stop'])))

f.close()