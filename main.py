# import matplotlib.pyplot as plt
# import numpy as np
# from scipy import stats
# import matplotlib.pyplot as plt
# from matplotlib import cm
# %matplotlib inline

# test using scikit-learn unit tests for linear classifier
# from sklearn.utils.estimator_checks import check_estimator
# check_estimator(EBLogisticRegression)
# check_estimator(VBLogisticRegression)
# from sklearn.datasets import make_blobs
# from matplotlib import cm
# from sklearn.cross_validation import train_test_split

from skbayes.linear_models import *
from sklearn.metrics import classification_report
from load_data import*
import numpy as np
from sklearn import metrics


#
# centers = [(-3, -3), (-3, 3), (3, 3)]
# n_samples = 600
#
# # create training & test set
# X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0, centers=centers, shuffle=False, random_state=42)
#
#
# X, x, Y, y = train_test_split(X, y, test_size=0.5, random_state=42)
#
# print(y)


train_image, train_labels, raw_train_labels, \
validation_image, valid_labels, raw_valid_labels, \
test_dataset, test_labels, raw_test_labels = import_MNIST()
print("MNIST Data fetched, done")
print('------------------------------')
validation_usps, validation_usps_label = load_usps('usps_test_image.pkl', 'usps_test_label.pkl')
print('usps validation_image: ', validation_usps.shape);
print('usps validation label: ', validation_usps_label.shape)
print("USPS Data fetched, done")
print('------------------------------')


x_train = train_image[0:50000]
y_train = raw_train_labels[0:50000]
x_test = validation_image[0:10000]
y_test = raw_valid_labels[0:10000]
x_valid = validation_image[0:10000];
y_valid = raw_valid_labels[0:10000];



#
eblr = EBLogisticRegression().fit(x_train, y_train);
vblr = VBLogisticRegression().fit(x_train, y_train);
eblr_prediction = eblr.predict(x_valid)
vblr_prediction = vblr.predict(x_valid)

#
print("\n === EBLogisticRegression on Validation set ===")
print(classification_report(y_valid, eblr_prediction))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_valid, eblr_prediction))


print("\n === VBLogisticRegression on Validation set  ===")
print(classification_report(y_valid, vblr_prediction))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_valid, vblr_prediction))




# fit rvc & svc
# vblr = VBLogisticRegression().fit(X, Y)
# eblr = EBLogisticRegression().fit(X, Y)




# # create grid
# n_grid = 100
# max_x = np.max(x, axis=0)
# min_x = np.min(x, axis=0)
# X1 = np.linspace(min_x[0], max_x[0], n_grid)
# X2 = np.linspace(min_x[1], max_x[1], n_grid)
# x1, x2 = np.meshgrid(X1, X2)
# Xgrid = np.zeros([n_grid ** 2, 2])
# Xgrid[:, 0] = np.reshape(x1, (n_grid ** 2,))
# Xgrid[:, 1] = np.reshape(x2, (n_grid ** 2,))
#
# eb_grid = eblr.predict_proba(Xgrid)
# vb_grid = vblr.predict_proba(Xgrid)
# grids = [eb_grid, vb_grid]
# names = ['EBLogisticRegression', 'VBLogisticRegression']
# # classes = np.unique(y)
#
# # # plot heatmaps
# # for grid, name in zip(grids, names):
# #     fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(20, 8))
# #     for ax, cl, model in zip(axarr, classes, grid.T):
# #         ax.contourf(x1, x2, np.reshape(model, (n_grid, n_grid)), cmap=cm.coolwarm)
# #         ax.plot(x[y == cl, 0], x[y == cl, 1], "ro", markersize=5)
# #         ax.plot(x[y != cl, 0], x[y != cl, 1], "bo", markersize=5)
# #     plt.suptitle(' '.join(['Decision boundary for', name, 'OVR multiclass classification']))
# #     plt.show()
#


# print("\n === EBLogisticRegression ===")
# print(classification_report(y, eblr.predict(x)))
# print("\n === VBLogisticRegression ===")
# print(classification_report(y, vblr.predict(x)))