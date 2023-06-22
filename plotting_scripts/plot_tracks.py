import numpy as np
import os
from pdb import set_trace as st
import matplotlib.pyplot as plt


# Paths to segmented track file
path_folder1 = r'Z:\Data\synchronised_imu_kinect_frames_no_interpolation'
path_folder2 = r'Z:\Data\synchronised_imu_kinect_frames'

# path1 = r'Z:\Data\synchronised_imu_kinect_frames_no_interpolation\20230522_165057 - 015 - KinectDataFrames.npy'
# path2 = r'Z:\Data\synchronised_imu_kinect_frames\20230519_170143 - 015 - KinectDataFrames.npy'

# List all Kinect frame files
list1 = os.listdir(path_folder1)
list2 = os.listdir(path_folder2)
list1 = [e for e in list1 if ' - KinectDataFrames.npy' in e]
list2 = [e for e in list2 if ' - KinectDataFrames.npy' in e]

# Loop on the listed data files
for file_idx in range(len(list1)):
    # Get current file
    file1 = list1[file_idx]
    # Extract subject idx
    pos = file1.find(' - KinectDataFrames.npy')
    subject_idx = file1[pos-3:pos]
    print('Checking Kinect tracks of subject %s' % (subject_idx))
    # Get corresponding file in the second folder
    corr_file = [e for e in list2 if subject_idx+' - KinectDataFrames.npy' in e]
    file2 = corr_file[0]
    # Load data frames
    f1 = np.load(os.path.join(path_folder1,file1))
    f2 = np.load(os.path.join(path_folder2,file2))

    # Look for differences in frames between f1 and f2
    for idx in range(len(f1)):
        tmp1 = f1[idx]
        tmp2 = f2[idx]
        if not (tmp1==tmp2).all():
            print('    Differences in frame %d/%d' % (idx+1,len(f1)))
            diff = tmp1-tmp2
            for column_idx in range(diff.shape[1]):
                # plot the two tracks
                if np.sum(diff[:,column_idx]) != 0:
                    plt.subplot(2,1,1)
                    plt.plot(tmp1[:,column_idx],color='b')
                    plt.subplot(2,1,2)
                    plt.plot(tmp2[:,column_idx],color='r')
                    plt.show()
