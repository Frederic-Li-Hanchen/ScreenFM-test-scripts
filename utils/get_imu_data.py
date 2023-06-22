import numpy as np
import py7zr
import os
import shutil
from pdb import set_trace as st
from time import time


###############################################################################################################
### Function to extract the IMU files from 7z archives and save them at a specified location.
###############################################################################################################
def copyImuFiles(
    dataPath, # Path to the folder containing the data files in 7z format. NOTE: works with either local or network drive paths on Windows.
    subjectId, # ID of the subject to get data from.
    targetFolder, # Path where the IMU files should be copied.
    encryptionPassword, # Password used to encrypt the 7z data.
    tmpFolder=None, # Folder where the data should be temporarily extracted. If set to None (default), data are extracted in the current folder. Data are automatically deleted after .csv file extraction
    ):

    print('Retrieving the IMU data for subject %d ...' % subjectId)

    # Path to the file
    pathToArchive = dataPath + '/' + str(subjectId).zfill(3) + '.7z'

    # Access archived contents by decompressing the archive
    # Note: can take between 20-30 minutes depending on the archive size
    start = time()
    with py7zr.SevenZipFile(pathToArchive,mode='r',password=encryptionPassword) as z:
        z.extractall(path=tmpFolder)
    end = time()
    print('Archive extracted in %.2f seconds' % (end-start))

    # Look for the IMU data
    if tmpFolder is not None:
        pathToExtractedFiles = tmpFolder+'/'+str(subjectId).zfill(3)+'/Recordings/'
    else:
        pathToExtractedFiles = './'+str(subjectId).zfill(3)+'/Recordings/'

    sessionNames = list(os.listdir(pathToExtractedFiles))
    sessionNames.sort()

    if len(sessionNames) == 2: # The first recording session is only for centring the iPhone, so discarded.
        # Locate the IMU files (.csv format)
        dataFiles = [e for e in os.listdir(pathToExtractedFiles+sessionNames[1]) if '.csv' in e]
        # Copy the IMU files (data and comments) to target location
        shutil.copy(pathToExtractedFiles+sessionNames[1]+'/'+dataFiles[0],targetFolder)
        shutil.copy(pathToExtractedFiles+sessionNames[1]+'/'+dataFiles[1],targetFolder)

    elif len(sessionNames) == 1:
        # Locate the IMU files (.csv format)
        dataFiles = [e for e in os.listdir(pathToExtractedFiles+sessionNames[0]) if '.csv' in e]
        # Copy the IMU files (data and comments) to target location
        shutil.copy(pathToExtractedFiles+sessionNames[0]+'/'+dataFiles[0],targetFolder)
        shutil.copy(pathToExtractedFiles+sessionNames[0]+'/'+dataFiles[1],targetFolder)

    else:
        print('    Incorrect number of IMU sessions retrieved! Subject %d ignored.' % subjectId)

    # Remove decompressed file
    if tmpFolder is not None:
        shutil.rmtree(tmpFolder+'/'+str(subjectId).zfill(3))
    else:
        shutil.rmtree('./'+str(subjectId).zfill(3))



###############################################################################################################
### Main
### NOTE: change parameters accordingly here
###############################################################################################################
if __name__ == '__main__':

    # Indices of the subjects to retrieve
    subjectId = [3]

    for subject in subjectId:
        copyImuFiles(
        dataPath='Z:/Data/raw/RealData/',
        #dataPath='./data/',
        subjectId=subject, # ID of the subject to get data from
        targetFolder='./target/', # Path where the IMU files should be copied
        encryptionPassword='U9X]{jj#^m', # Password used to encrypt the 7z data
        tmpFolder='./test/', # Folder where the data should be temporarily extracted. If set to None (default), data are extracted in the current folder
        )