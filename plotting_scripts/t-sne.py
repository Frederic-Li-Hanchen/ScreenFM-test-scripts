import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

'''
Script that runs t-SNE for given numpy files!
'''

#TODO: This script could be adapted to perfom t-SNE automatically for all FOLDS!


# features_train = np.load('../../data/dnn_features/DISFM_v2_features_train_fold_1.npy')
# labels_train = np.load('../../data/dnn_features/DISFM_v2_labels_train_fold_1.npy')
# id_labels_train = np.load('../../data/dnn_features/DISFM_v2_id_labels_train_fold_1.npy')
# train_zero = np.count_nonzero(features_train[0]==0)

# features_test = np.load('../../data/dnn_features/DISFM_v2_features_test_fold_1.npy')
# labels_test = np.load('../../data/dnn_features/DISFM_v2_labels_test_fold_1.npy')
# id_labels_test = np.load('../../data/dnn_features/DISFM_v2_id_labels_test_fold_1.npy')
# test_zero = np.count_nonzero(features_test[0]==0)

features_train = np.load('./dnn_features/train_x_fold1.npy')
labels_train = np.load('./dnn_features/train_y_fold1.npy')
id_labels_train = np.load('./dnn_features/train_subject_id_fold1.npy')
train_zero = np.count_nonzero(features_train[0]==0)

features_test = np.load('./dnn_features/test_x_fold1.npy')
labels_test = np.load('./dnn_features/test_y_fold1.npy')
id_labels_test = np.load('./dnn_features/test_subject_id_fold1.npy')
test_zero = np.count_nonzero(features_test[0]==0)

transformed_features_train = TSNE().fit_transform(X=features_train)
transformed_features_test = TSNE().fit_transform(X=features_test)


tsne_data_frame_train = pd.DataFrame({'labels': labels_train,
                                      'feature_1': transformed_features_train[:, 0],
                                      'feature_2': transformed_features_train[:, 1]})
tsne_data_frame_test = pd.DataFrame({'labels': labels_test,
                                     'feature_1': transformed_features_test[:, 0],
                                     'feature_2': transformed_features_test[:, 1]})

plt.figure(figsize=(14, 14))
plt.title('Visualization of t-SNE Train results on SFM Dataset', fontsize=24, weight='bold')
sns.scatterplot(data=tsne_data_frame_train, x='feature_1', y='feature_2', hue='labels',
                legend='full', palette='Set1')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Feature_1')
plt.ylabel('Feature_2')
plt.legend(fontsize=16)
plt.show()


plt.figure(figsize=(14, 14))
plt.title('Visualization of t-SNE Test results on SFM Dataset', fontsize=24, weight='bold')
sns.scatterplot(data=tsne_data_frame_test, x='feature_1', y='feature_2', hue='labels',
                legend='full', palette='Set1')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Feature_1', fontsize=16)
plt.ylabel('Feature_2', fontsize=16)
plt.legend(fontsize=16)
plt.show()
