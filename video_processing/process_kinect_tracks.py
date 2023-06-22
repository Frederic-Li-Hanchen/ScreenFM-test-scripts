import numpy as np
import os
from pdb import set_trace as st
import pandas as pd
from math import floor, ceil
from datetime import datetime

# # Path to the IMU file
# imu_path = './data/001 – 20220120T083609Z – IMU.csv'

# Path to the track file
track_path = './video_processing/kinect_landmarks/'
track_name = '20220120_090931_001.more_lm_dep3d'
pos = track_name.find('.more_lm_dep3d')
subject_id = track_name[pos-3:pos]

track_path2 = './video_processing/kinect_landmarks/20220120_090931_001.more_lm_col'
track_data2 = pd.read_csv(track_path2)

# Path to the offset file
offset_path = './offsets_df.xlsx'

# Segmentation window size
fps = 30 # Obtained by checking the average difference between two consecutive timestamps in the track file (33.3ms, i.e. 30 frames per second)
window_size = 30 # 1 second of data since fps=30

# Save path
save_path = './video_processing/segmented_tracks/'

# # Load IMU data
# imu_data = pd.read_csv(imu_path)
# LH_imu_data = imu_data.loc[imu_data['name']=='LH']

# Load track data and remove 'confidence' columns
track_data = pd.read_csv(os.path.join(track_path,track_name))
columns_to_keep = [e for e in track_data.columns if 'confidence' not in e]
track_data = track_data[columns_to_keep]

#st()

# Load offsets
imu_kinect_sync = pd.read_excel(offset_path)
pos = track_name.find('.more_lm_dep3d')
file_name = track_name[:pos]
b = imu_kinect_sync['Filename Kinect'].str.contains(file_name)
subject_info = imu_kinect_sync.loc[b]
offset = subject_info['Offset Time [s]'].iloc[0] # in s
frame_offset = subject_info['Offset Frames'].iloc[0] # in frames

# Align the Kinect data stream with the iPhone/IMU stream
# There are two cases to consider:
#   1- offset>0: the Kinect started before the iPhone. In that case, N frames are removed from the Kinect stream before segmentation, where N is the 'Offset Frames' information
#   2- offset<0: the iPhone started before the Kinect. In that case, N frames should be removed from the IMU stream where N is the 'Offset Frames' information.
#      Because the IMU data was already segmented in frames of 1 second, the Kinect stream is aligned with a second-wise granularity to the start of the iPhone stream
#      instead for further alignement by removing 1-second frames from the IMU stream.
if offset >= 0: # Kinect started before iPhone
    nb_frames_to_discard = frame_offset
else: # iPhone started before Kinect
    time_shift = ceil(offset) - offset
    nb_frames_to_discard = ceil(time_shift*fps)

track_data = track_data.iloc[nb_frames_to_discard:]

# Loop on the data to segment it
nb_timestamps = len(track_data)
nb_segments = floor(len(track_data)/window_size)
nb_coordinates = len(track_data.columns)-1 # Removal of timestamp column
point_coordinates = [e for e in track_data.columns if 'timestamp' not in e]
segmented_frames = np.zeros((nb_segments,window_size,nb_coordinates),dtype=np.float32)
running_frame_idx = 0

while running_frame_idx < nb_segments:
    segmented_frames[running_frame_idx] = track_data[point_coordinates].iloc[running_frame_idx*window_size:(running_frame_idx+1)*window_size].to_numpy()
    running_frame_idx += 1

# Save the created numpy array of segmented frames
current_time = datetime.now()
file_name = str(current_time.year)+str(current_time.month).zfill(2)+str(current_time.day).zfill(2)+'_' \
    + str(current_time.hour).zfill(2)+str(current_time.minute).zfill(2)+str(current_time.second).zfill(2) \
    + ' - ' + subject_id + ' - KinectDataFrames.npy'

np.save(os.path.join(save_path,file_name),segmented_frames)