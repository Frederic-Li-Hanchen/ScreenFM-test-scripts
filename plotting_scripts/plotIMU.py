import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pdb import set_trace as st
import os
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 7


# Import the data
subjectId = 49
dataPath = r'.\data\\'
fileList = os.listdir(dataPath)
fileList = [e for e in fileList if str(subjectId).zfill(3)+' – ' in e and 'IMU.csv' in e]
dataFilePath = os.path.join(dataPath,fileList[0])

data = pd.read_csv(dataFilePath)

print('Data contains NaN values: %r' % data.isnull().values.any())

# Separate data per device
deviceNames = list(set(data['name']))
deviceNames.sort()
nbDevices = len(deviceNames)

deviceData = {}
for idx in range(nbDevices):
    deviceData[deviceNames[idx]] = data[data['name']==deviceNames[idx]]

# Plot the data for each device separately
#fig, ax = plt.subplots(nbDevices,3)
fig, ax = plt.subplots(nbDevices,2)
fig.tight_layout()

#plt.setp(ax, xticks=[], yticks=[])

for idx in range(nbDevices):
    currentData = deviceData[deviceNames[idx]]
    # Limit the number of points to plot for clarity
    #currentData = currentData.iloc[0:10000]
    timestamps = currentData['timestamp']
    ax[idx,0].plot(timestamps,currentData['accx'],'r')
    ax[idx,0].plot(timestamps,currentData['accy'],'b')
    ax[idx,0].plot(timestamps,currentData['accz'],'g')
    ax[idx,0].set_title(deviceNames[idx]+' acc')
    ax[idx,1].plot(timestamps,currentData['gyrox'],'r')
    ax[idx,1].plot(timestamps,currentData['gyroy'],'b')
    ax[idx,1].plot(timestamps,currentData['gyroz'],'g')
    ax[idx,1].set_title(deviceNames[idx]+' gyro')

    # ax[idx,0].xaxis.set_ticklabels([])
    # ax[idx,0].xaxis.set_ticks_position('none')
    # ax[idx,0].yaxis.set_ticklabels([])
    # ax[idx,0].yaxis.set_ticks_position('none')
    # ax[idx,1].xaxis.set_ticklabels([])
    # ax[idx,1].xaxis.set_ticks_position('none')
    # ax[idx,1].yaxis.set_ticklabels([])
    # ax[idx,1].yaxis.set_ticks_position('none')

    #ax[idx,2].plot(timestamps,currentData['magnx'],'r')
    #ax[idx,2].plot(timestamps,currentData['magny'],'b')
    #ax[idx,2].plot(timestamps,currentData['magnz'],'g')
    # print('######### '+deviceNames[idx] + ' ##########')
    # print(np.std(currentData['magnx']))
    # print(np.std(currentData['magny']))
    # print(np.std(currentData['magnz']))
    #ax[idx,2].set_title(deviceNames[idx]+' magn')


plt.show()
