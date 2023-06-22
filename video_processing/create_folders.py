import numpy as np
from pdb import set_trace as st
import os
import shutil

# Path to the folder
folder_path = r'D:\\'

# List comment and video files inside folder
comment_files = [e for e in os.listdir(folder_path) if 'Comments.csv' in e]
video_files = [e for e in os.listdir(folder_path) if '.mkv.mp4' in e]

nb_subjects = len(comment_files)

for idx in range(1,nb_subjects+1):
    # Create target folder
    subject_id = str(idx).zfill(3)
    os.mkdir(folder_path+subject_id)
    # Look for the relevant files
    tmp_list = [e for e in comment_files if subject_id+' – ' in e]
    comments = tmp_list[0]
    tmp_list = [e for e in video_files if '_'+subject_id+'.mkv.mp4' in e]
    video = tmp_list[0]
    # Move the files in the target folder
    shutil.move(folder_path+comments,folder_path+subject_id+r'\\'+comments)
    shutil.move(folder_path+video,folder_path+subject_id+r'\\'+video)