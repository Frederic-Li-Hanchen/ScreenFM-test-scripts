import numpy as np
from pdb import set_trace as st
import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
#from itertools import permutations
from sklearn.metrics import accuracy_score, f1_score


# Define the folds
fold_1 = ['001', '007', '012', '018', '023', '028', '034', '039', '044', '052']
fold_2 = ['002', '008', '013', '019', '024', '029', '035', '040', '045', '068']
fold_3 = ['004', '009', '015', '020', '025', '030', '036', '041', '046', '031']
fold_4 = ['005', '010', '016', '021', '026', '032', '037', '042', '047', '048']
fold_5 = ['006', '011', '017', '022', '027', '033', '038', '043', '049']
nb_folds = 5
accuracies = np.zeros(nb_folds)
af1 = np.zeros(nb_folds)

# Data path
data_folder_path = './data/'

# Define CNN architecture
def cnn(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(16,(3,1),activation='relu')(inputs)
    x = BatchNormalization(axis=2)(x)
    x = MaxPooling2D((2,1))(x)
    x = Conv2D(32,(3,1),activation='relu')(x)
    x = BatchNormalization(axis=2)(x)
    x = MaxPooling2D((2,1))(x)
    x = Conv2D(64,(3,1),activation='relu')(x)
    x = BatchNormalization(axis=2)(x)
    x = Flatten()(x)
    x = Dense(100,activation='tanh')(x)
    x = Dropout(0.1)(x)
    x = Dense(50,activation='tanh')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(2,activation='softmax')(x)
    model = Model(inputs=inputs,outputs=outputs)
    return model

# Loop on the folds
for idx in range(nb_folds):

    print('')
    print('#################################################')
    print('Processing fold %d/%d ...' % (idx+1,nb_folds))

    # Define the dataset indices
    if idx == 0:
        train_ids = fold_1 + fold_2 + fold_3 + fold_4
        test_ids = fold_5
    elif idx == 1:
        train_ids = fold_1 + fold_2 + fold_3 + fold_5
        test_ids = fold_4
    elif idx == 2:
        train_ids = fold_1 + fold_2 + fold_4 + fold_5
        test_ids = fold_3
    elif idx == 3:
        train_ids = fold_1 + fold_3 + fold_4 + fold_5
        test_ids = fold_2
    else:
        train_ids = fold_2 + fold_3 + fold_4 + fold_5
        test_ids = fold_1

    # Prepare the data and labels for the training and testing set
    list_files = os.listdir(data_folder_path)
    kinect_files = [e for e in list_files if 'KinectDataFrames.npy' in e]
    label_files = [e for e in list_files if 'Labels.npy' in e]
    train_kinect_files = []
    train_label_files = []
    test_kinect_files = []
    test_label_files = []
    for file_name in kinect_files:
        # extract subject idx
        pos = file_name.find(' - KinectDataFrames.npy')
        index = file_name[pos-3:pos]
        # look for corresponding label file
        current_label = [e for e in label_files if ' - '+index+' - Labels.npy' in e]
        if index in train_ids:
            train_kinect_files += [file_name]
            train_label_files += current_label
        else:
            test_kinect_files += [file_name]
            test_label_files += current_label

    # Determine the training and testing set sizes
    train_size = 0
    test_size = 0
    for file_name in train_kinect_files:
        tmp_data = np.load(os.path.join(data_folder_path,file_name))
        nb_examples, time_window, nb_channels = tmp_data.shape
        train_size += nb_examples
    train_data = np.zeros((train_size,time_window,nb_channels),dtype=np.float32)
    train_labels = np.zeros((train_size),dtype=int)
    for file_name in test_kinect_files:
        tmp_data = np.load(os.path.join(data_folder_path,file_name))
        nb_examples, time_window, nb_channels = tmp_data.shape
        test_size += nb_examples
    test_data = np.zeros((test_size,time_window,nb_channels),dtype=np.float32)
    test_labels = np.zeros((test_size),dtype=int)

    # Load the data
    current_idx = 0
    for file_name in train_kinect_files:
        tmp_data = np.load(os.path.join(data_folder_path,file_name))
        train_data[current_idx:current_idx+len(tmp_data)] = tmp_data
        current_idx += len(tmp_data)
    current_idx = 0
    for file_name in train_label_files:
        tmp_data = np.load(os.path.join(data_folder_path,file_name))
        train_labels[current_idx:current_idx+len(tmp_data)] = tmp_data
        current_idx += len(tmp_data)
    current_idx = 0
    for file_name in test_kinect_files:
        tmp_data = np.load(os.path.join(data_folder_path,file_name))
        test_data[current_idx:current_idx+len(tmp_data)] = tmp_data
        current_idx += len(tmp_data)
    current_idx = 0
    for file_name in test_label_files:
        tmp_data = np.load(os.path.join(data_folder_path,file_name))
        test_labels[current_idx:current_idx+len(tmp_data)] = tmp_data
        current_idx += len(tmp_data)

    # Remove examples associated with invalid data
    valid_train_idx = [i for i,e in enumerate(train_labels) if e!=-1]
    train_data = train_data[valid_train_idx]
    train_labels = train_labels[valid_train_idx]
    valid_test_idx = [i for i,e in enumerate(test_labels) if e!=-1]
    test_data = test_data[valid_test_idx]
    test_labels = test_labels[valid_test_idx]

    # Replace NaN to 0
    np.nan_to_num(train_data,copy=False,nan=0)
    np.nan_to_num(test_data,copy=False,nan=0)

    # Shuffle datasets in unisson
    perm = np.random.permutation(len(train_data))
    train_data = train_data[perm]
    train_labels = train_labels[perm]
    perm = np.random.permutation(len(test_data))
    test_data = test_data[perm]
    test_labels = test_labels[perm]

    # Prepare input data
    input_shape = (train_data[0].shape[0],train_data[0].shape[1],1)
    model = cnn(input_shape=input_shape)
    train_data = np.expand_dims(train_data,-1)
    test_data = np.expand_dims(test_data,-1)
    train_labels_cat = tf.keras.utils.to_categorical(train_labels,num_classes=2)
    test_labels_cat = tf.keras.utils.to_categorical(test_labels,num_classes=2)

    # Train the CNN
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )

    #model.summary()

    model.fit(
        train_data, 
        train_labels_cat,
        batch_size=16,
        epochs=1,
        validation_data=(test_data,test_labels_cat)
    )

    # Evaluate the CNN
    softmax_scores = model.predict(test_data)
    estimations = np.argmax(softmax_scores,axis=1)
    accuracies[idx] = 100*accuracy_score(test_labels,estimations)
    af1[idx] = 100*f1_score(test_labels,estimations,average='macro')

    # Clear session
    tf.keras.backend.clear_session()

# Print results
print('')
print('#################################################')
print('Average accuracy over %d folds: %.2f +- %.2f %%' % (nb_folds, np.mean(accuracies), np.std(accuracies)))
print('Average AF1 over %d folds: %.2f +- %.2f %%' % (nb_folds, np.mean(af1), np.std(af1)))
print('#################################################')