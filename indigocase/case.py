# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# Your objective is to use any method you want to develop an automatic pipeline with the training data (“train.tif” and “train_ground_truth.shp”) and to count the number of plants in the “test.tif” file as accurately as possible. 
#
# However, building a highly accurate model is only part of the story. We also want to know how you do it. As such, we’ll also be looking for you to deliver a 5-page slides on your approach. Additionally, we’d like you to put your codes in a zip file and send to us which should to include python files (.py) that can be run in command line and a notebook file (.ipynb) to walk through the pipeline. 
#
# The final component of the evaluation is to generate georeferenced maps using the stand count results from your automatic pipeline: a plant population map (the unit for the pixel is plants/ac) and a plant size map (the unit for the pixel is cm).  

# +
import matplotlib
new_style = {'grid': False}
matplotlib.rc('axes', **new_style)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import shape
from osgeo import gdal, osr

import geopandas as gpd

# -

# a TIFF file: “train.tif”. 
# - This is an RGB map over a half of a corn field. The corn plants are at V2-3 growth stage. 
#
#  
# a TIFF file: “test.tif”.
# - This is an RGB map over the other half of the corn field. The corn plants are at V2-3 growth stage. 
#
#  
# a shapefile: “train_ground_truth.shp” and its related files (you will need to download all of them to view the shapefile).
# - This is the ground truth data for the stand count case study, which can be used to develop and evaluate your automatic stand counting method. You can open in QGIS or any GIS software. Each point in the file represents the location of an individual plant. 

# +
# def fnames
import os
dpath = 'data/'
trainf = os.path.join(dpath, 'train.tif')
testf = os.path.join(dpath, 'test.tif')
truthf = os.path.join(dpath, 'train_ground_truth.shp')

assert os.path.isfile(trainf)
assert os.path.isfile(testf)
assert os.path.isfile(truthf)
# -

truth = gpd.read_file(truthf)

truth.columns

len(truth)

truth.geometry[:5]

# https://www.spatialreference.org/ref/epsg/4326/
truth.crs

# + {"active": ""}
# (base) dleung@C02X61QTJHD5:~/Hack/GIS/Indigo_Oct/data (master *)$ gdalinfo test.tif 
# Driver: GTiff/GeoTIFF
# Files: test.tif
# Size is 6720, 4865
# Coordinate System is:
# GEOGCS["WGS 84",
#     DATUM["WGS_1984",
#         SPHEROID["WGS 84",6378137,298.257223563,
#             AUTHORITY["EPSG","7030"]],
#         AUTHORITY["EPSG","6326"]],
#     PRIMEM["Greenwich",0],
#     UNIT["degree",0.0174532925199433],
#     AUTHORITY["EPSG","4326"]]
# Origin = (-89.812908406592086,35.125241201422014)
# Pixel Size = (0.000000072319788,-0.000000072319788)
# Metadata:
#   acquisitionEndDate=2019-06-28T13:20:24+00:00
#   acquisitionStartDate=2019-06-28T12:50:24+00:00
#   AREA_OR_POINT=Area
#   isCalibrated=False
# Image Structure Metadata:
#   INTERLEAVE=PIXEL
# pj_obj_create: Cannot find proj.db
# Corner Coordinates:
# Upper Left  ( -89.8129084,  35.1252412) ( 89d48'46.47"W, 35d 7'30.87"N)
# Lower Left  ( -89.8129084,  35.1248894) ( 89d48'46.47"W, 35d 7'29.60"N)
# Upper Right ( -89.8124224,  35.1252412) ( 89d48'44.72"W, 35d 7'30.87"N)
# Lower Right ( -89.8124224,  35.1248894) ( 89d48'44.72"W, 35d 7'29.60"N)
# Center      ( -89.8126654,  35.1250653) ( 89d48'45.60"W, 35d 7'30.24"N)
# Band 1 Block=6720x1 Type=Byte, ColorInterp=Red
#   Mask Flags: PER_DATASET ALPHA 
# Band 2 Block=6720x1 Type=Byte, ColorInterp=Green
#   Mask Flags: PER_DATASET ALPHA 
# Band 3 Block=6720x1 Type=Byte, ColorInterp=Blue
#   Mask Flags: PER_DATASET ALPHA 
# Band 4 Block=6720x1 Type=Byte, ColorInterp=Alpha

# +
datasetTrain = rasterio.open(trainf)
print(datasetTrain.meta)

datasetTest = rasterio.open(testf)
print(datasetTest.meta)
# -

# No data values for all channels
print(datasetTrain.nodatavals)
print(datasetTest.nodatavals)

# +
np.seterr(divide='ignore', invalid='ignore')

def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)


# -

with rasterio.open(trainf) as src:
    profile = src.profile
    print(src.profile)
    
    (xmin, ymin, xmax, ymax) = src.bounds
    width = src.width
    height = src.height
#    print(src.descriptions)
    train_img1 = src.read(1).astype(float)
    train_img2 = src.read(2).astype(float)
    train_img3 = src.read(3).astype(float)
    train_img4 = src.read(4).astype(float)

    # Normalize band 
    train_img4_norm = normalize(train_img4)
    train_img3_norm = normalize(train_img3)
    train_img2_norm = normalize(train_img2)
    train_img1_norm = normalize(train_img1)    

# +
# Stack bands
false_RGB = np.dstack((train_img4_norm, train_img3_norm, train_img2_norm))

plt.hist(false_RGB.flatten())
plt.show();
# -

# View the color composite
plt.imshow(false_RGB)
plt.show();

# +
from skimage import exposure
p5, p98 = np.percentile(false_RGB, (5, 98))                             
rgbStretched = exposure.rescale_intensity(false_RGB, in_range=(p5, p98)) # Perform contrast stretch on RGB range
rgbStretched = exposure.adjust_gamma(rgbStretched, 0.5)                  # Perform Gamma Correctio

fig = plt.figure()
ax = plt.Axes(fig,[0,0,1,1]) 
ax.set_axis_off()                                            # Turn off axes
fig.add_axes(ax)
ax.imshow(rgbStretched, interpolation='bilinear', alpha=0.9) # Plot a natural color RGB
plt.show();
# -

# read all bands and compute summary stats
TrainALLbands = datasetTrain.read()
stats = [] 
for band in TrainALLbands:
     stats.append({
         'min': band.min(),
         'mean': band.mean(),
         'median': np.median(band),
         'max': band.max()})
print(stats)

# plot train
from rasterio.plot import show
show(datasetTrain)
plt.savefig('train_basic.png', bbox_inches="tight")

from rasterio.plot import show_hist
show_hist(datasetTrain, bins=50, lw=0.0, stacked=False, alpha=0.3,
        histtype='stepfilled', title="Histogram")


# +
# ndvi
nir = train_img4
red = train_img1

def ndviCalc(red, nir):
    return ((nir - red)/(nir + red))

ndvi = ndviCalc(red, nir)
print(np.isnan(ndvi).sum())
ndvi[np.where(np.logical_or(ndvi < 0, ndvi > 1)) ] = np.nan 
print(np.isnan(ndvi).sum())
print(np.nanmean(ndvi))
print(np.nanstd(ndvi))
show(ndvi, cmap='summer')
# -

train_img4.type()

show_hist(ndvi, bins=50, lw=0.0, stacked=False, alpha=0.3,
        histtype='stepfilled', title="Histogram on ndvi")
plt.show()

truth.shape

fig, ax = plt.subplots(figsize = (10,10))
truth.plot(ax=ax)
print(xmin, ymin, xmax, ymax)
plt.ylim([35.125, 35.1259])
plt.xlim([-89.813, -89.8123])
ylim = plt.gca().get_ylim()
xlim = plt.gca().get_xlim()
plt.show()

fig, ax = plt.subplots(figsize = (10,10))
plt.xlim(xlim)
plt.ylim(ylim)
show(datasetTrain)

fig, ax = plt.subplots(figsize = (10,10))
plt.xlim(xlim)
plt.ylim(ylim)
show(datasetTest)

# +
fig, ax = plt.subplots()
show((datasetTrain, 1), ax=ax)
truth.plot(ax=ax, color='red', alpha=0.05)

plt.title('Truth overplotted on Train field')
plt.show()
# -

# ### ML

# +
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors
from skimage import feature, filters
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
# -

# ### classification on pixels in ground truth 
# Train: 
# - use pixels as knn train
#
# Test:
# - predict pixels on train image
# - predict pixels on test image, but we would have to know ytest (which I could "generate" using subset of the training data)
#
# Predict:
# - run model on the test field --> # of plants

# remove rows where ndvi is nan, since ML can train on those rows
badii = np.isnan(ndvi)
print(badii.sum(), len(ndvi.flatten()))

# +
# build ytarget based on ground truth
ytarget = np.zeros_like(train_img1)
import fiona
with fiona.open(truthf, "r") as shfile:
    features = [feature['geometry'] for feature in shfile]
    
for i in features:
    # get the x,y pix position of each corn pos 
    py, px = datasetTrain.index(*i['coordinates'])
    # print('Pixel Y, X coords: {}, {}'.format(py, px))
    ytarget[py, px] = 1.0
ytarget = ytarget[~badii].flatten().astype(int)
# -

# build DF as X
X_train = pd.DataFrame()
X_train['b1'] = train_img1[~badii].flatten()
X_train['b2'] = train_img2[~badii].flatten()
X_train['b3'] = train_img3[~badii].flatten()
X_train['b4'] = train_img4[~badii].flatten()
X_train['ndvi'] = ndvi[~badii].flatten()
cnames = X_train.columns.values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns=cnames)
X_train.describe()

assert ytarget.shape[0] == X_train.shape[0]

# +
import copy
# Visualization of attributes/features with output
from pandas.plotting import scatter_matrix
import random

from matplotlib import cm
cmap = cm.get_cmap('gnuplot')
_X_train_tmp = X_train.copy(deep=True)
_X_train_tmp['output'] = ytarget
ax = scatter_matrix(_X_train_tmp.sample(n=30000), alpha=0.2, figsize=(15, 15),
                        hist_kwds={'bins': 35}) #,
                        # diagonal='kde')
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.2, left=0.2)
plt.suptitle('Scatter-matrix for each input variable')
plt.show()
# -

sns.countplot(x='output', data=_X_train_tmp, palette='hls')
plt.show()
print(_X_train_tmp['output'].sum()/len(_X_train_tmp['output']))

# ### Split the training data so I could calculate test scores

X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(X_train, ytarget, test_size=0.3)

# +
### 0 and 1, so start w/ logistic 
logreg = LogisticRegression()
logreg.fit(X_train_small, y_train_small)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train_small, y_train_small)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test_small, y_test_small)))

# this is probably because we have very imbalanced data.. most of the pixels are not plants

# +
### 0 and 1, so start w/ logistic 
logreg = LogisticRegression(class_weight='balanced')
logreg.fit(X_train_small, y_train_small)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train_small, y_train_small)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test_small, y_test_small)))

# -

# ### Tune hyperparameters? and CV
# * Regularization using L2 to prevent terms from going to zero in L1
#
# Deal with imbalance data
# * upweight
# * “Calibration”

# gridsearch 
dual = [True, False]
max_iter = [100, 120, 140]
# C = [1.0,1.5,2.0,2.5]
param_grid = dict(dual=dual, max_iter=max_iter) #, C=C)

# +
# logreg_tuned = LogisticRegression(penalty='l2', class_weight='balanced')
# grid = GridSearchCV(estimator=logreg_tuned, param_grid=param_grid, cv=3, n_jobs=-1)
# grid_result = grid.fit(X_train_small, y_train_small)
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# +
# logreg_tuned = LogisticRegression(penalty='l2', **grid_result.best_params_)
# logreg_tuned.fit(X_train_small, y_train_small)
                                  
# print('Accuracy of Logistic regression classifier on training set: {:.2f}'
#      .format(logreg_tuned.score(X_train_small, y_train_small)))
# print('Accuracy of Logistic regression classifier on test set: {:.2f}'
#      .format(logreg_tuned.score(X_test_small, y_test_small)))
                                  
# -

import sklearn.metrics as metrics
metrics.SCORERS.keys()

# +
# stability using kCV, metric using accuracy
clf = logreg      # logreg_tuned

kfold = KFold(n_splits=3, random_state=7)
result = cross_val_score(clf, X_train, ytarget, cv=kfold, scoring='accuracy')
print(result.mean())
# -

# Re-evaluate w/ diff. metric due to imbalanced data
result = cross_val_score(clf, X_train, ytarget, cv=kfold, scoring='balanced_accuracy')
print(result.mean())

print(y_test_small.sum())
print(len(y_test_small) - y_test_small.sum())

# +
import matplotlib.pyplot as plt
import seaborn as sns

y_test_pred = clf.predict(X_test_small)
confusion = confusion_matrix(y_test_small, y_test_pred)
print(confusion)

sns.heatmap(confusion, annot=True, cbar=False)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print(classification_report(y_test_small, y_test_pred))
# -

# ### Read in test field data to calc number of obj

# +
with rasterio.open(testf) as src:
    profile = src.profile
    print(src.profile)
    
    (xmin, ymin, xmax, ymax) = src.bounds
    width = src.width
    height = src.height
#    print(src.descriptions)
    test_img1 = src.read(1).astype(float)
    test_img2 = src.read(2).astype(float)
    test_img3 = src.read(3).astype(float)
    test_img4 = src.read(4).astype(float)

    # Normalize band 
    test_img4_norm = normalize(test_img4)
    test_img3_norm = normalize(test_img3)
    test_img2_norm = normalize(test_img2)
    test_img1_norm = normalize(test_img1)    

red = test_img1
nir = test_img4
ndviTest = ndviCalc(red, nir)
print(np.isnan(ndviTest).sum())
ndvi[np.where(np.logical_or(ndviTest < 0, ndviTest > 1)) ] = np.nan 
print(np.isnan(ndviTest).sum())
print(np.nanmean(ndviTest))
print(np.nanstd(ndviTest))
show(ndviTest, cmap='summer')
# -

# remove rows where ndviTest is nan, since ML momdel not trained for those pixels
badiiTest = np.isnan(ndviTest)
print(badiiTest.sum(), len(ndviTest.flatten()))

# build DF as X
XXX = pd.DataFrame()
XXX['b1'] = test_img1[~badiiTest].flatten()
XXX['b2'] = test_img2[~badiiTest].flatten()
XXX['b3'] = test_img3[~badiiTest].flatten()
XXX['b4'] = test_img4[~badiiTest].flatten()
XXX['ndvi'] = ndviTest[~badiiTest].flatten()
cnames = XXX.columns.values

# generate X for the test field
XXX = scaler.fit_transform(XXX)
XXX = pd.DataFrame(XXX, columns=cnames)
XXX.describe()

# run model on the test field
yyy_predict_logreg = clf.predict(XXX)

print(yyy_predict_logreg.sum()/len(yyy_predict_logreg))
print(yyy_predict_logreg.sum())

# summarizes the model’s performance by evaluating the trade offs between 
# true positive rate (sensitivity) and false positive rate(1- specificity)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test_small, clf.predict(X_test_small))
fpr, tpr, thresholds = roc_curve(y_test_small, clf.predict_proba(X_test_small)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
# plt.savefig('Log_ROC')
plt.show()

# ### SVM Classifier 

# +
from sklearn.svm import SVC

SVM = SVC(kernel='linear', class_weight='balanced')
SVM.fit(X_train_small, y_train_small)

print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(SVM.score(X_train_small, y_train_small)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(SVM.score(X_test_small, y_test_small)))
# -
y_pred_small = SVM.predict(X_test_small)
print("Precision:",metrics.precision_score(y_test_small, y_pred_small))
print("Recall:",metrics.recall_score(y_test_small, y_pred_small))

# + {"endofcell": "--"}
clf = logreg      # logreg_tuned

kfold = KFold(n_splits=3, random_state=7)
result = cross_val_score(clf, X_train, ytarget, cv=kfold, scoring='accuracy')
print(result.mean())
# -

# Re-evaluate w/ diff. metric due to imbalanced data
result = cross_val_score(clf, X_train, ytarget, cv=kfold, scoring='balanced_accuracy')
print(result.mean())
# --

# +
y_test_pred = SVM.predict(X_test_small)
confusion = confusion_matrix(y_test_small, y_test_pred)
print(confusion)

sns.heatmap(confusion, annot=True, cbar=False)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print(classification_report(y_test_small, y_test_pred))
# -

# summarizes the model’s performance by evaluating the trade offs between 
# true positive rate (sensitivity) and false positive rate(1- specificity)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test_small, clf.predict(X_test_small))
fpr, tpr, thresholds = roc_curve(y_test_small, clf.predict_proba(X_test_small)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='SVM Linear (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# run model on the test field
yyy_predict_linearSVM = clf.predict(XXX)

# ### Tuning Hyperparameters (maybe?)
# - Kernel: The main function of the kernel is to transform the given dataset input data into the required form. There are various types of functions such as linear, polynomial, and radial basis function (RBF). Polynomial and RBF are useful for non-linear hyperplane. Polynomial and RBF kernels compute the separation line in the higher dimension. In some of the applications, it is suggested to use a more complex kernel to separate the classes that are curved or nonlinear. This transformation can lead to more accurate classifiers.
#
# - Regularization: Regularization parameter in python's Scikit-learn C parameter used to maintain regularization. Here C is the penalty parameter, which represents misclassification or error term. The misclassification or error term tells the SVM optimization how much error is bearable. This is how you can control the trade-off between decision boundary and misclassification term. A smaller value of C creates a small-margin hyperplane and a larger value of C creates a larger-margin hyperplane.
#
# - Gamma: (for non-linear). A lower value of Gamma will loosely fit the training dataset, whereas a higher value of gamma will exactly fit the training dataset, which causes over-fitting. In other words, you can say a low value of gamma considers only nearby points in calculating the separation line, while the a value of gamma considers all the data points in the calculation of the separation line.

# ### SMV: non-linear kernel, poly

nfolds = 3
Cs = [0.1, 1, 10]
degrees = [1, 2, 3]
gammas = [0.1, 1, 10]
param_grid = {'C': Cs, 'degree': degrees, 'gamma' : gammas}
grid_search = GridSearchCV(estimator=SVC(kernel='poly'), param_grid, cv=nfolds)
grid_search.fit(X_train, ytarget)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# +
SVM = SVC(kernel='poly', class_weight='balanced', **grid_result.best_params_)
SVM.fit(X_train_small, y_train_small)

print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(SVM.score(X_train_small, y_train_small)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(SVM.score(X_test_small, y_test_small)))

y_pred_small = SVM.predict(X_test_small)
print("Precision:",metrics.precision_score(y_test_small, y_pred_small))
print("Recall:",metrics.recall_score(y_test_small, y_pred_small))

# + {"endofcell": "--"}
clf = SVM    

kfold = KFold(n_splits=nfolds, random_state=7)
result = cross_val_score(clf, X_train, ytarget, cv=kfold, scoring='accuracy')
print(result.mean())
# -

# Re-evaluate w/ diff. metric due to imbalanced data
result = cross_val_score(clf, X_train, ytarget, cv=kfold, scoring='balanced_accuracy')
print(result.mean())
# --

# +
y_test_pred = clf.predict(X_test_small)
confusion = confusion_matrix(y_test_small, y_test_pred)
print(confusion)

sns.heatmap(confusion, annot=True, cbar=False)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print(classification_report(y_test_small, y_test_pred))
# -

# run model on the test field
yyy_predict_polySVM = clf.predict(XXX)

# summarizes the model’s performance by evaluating the trade offs between 
# true positive rate (sensitivity) and false positive rate(1- specificity)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test_small, clf.predict(X_test_small))
fpr, tpr, thresholds = roc_curve(y_test_small, clf.predict_proba(X_test_small)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='SVM Linear (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# ### SMV: non-linear kernel, rbf

nfolds = 3
Cs = [0.1, 1, 10]
gammas = [0.1, 1, 10]
param_grid = {'C': Cs, 'gamma' : gammas}
grid_search = GridSearchCV(estimator=SVC(kernel='rbf'), param_grid, cv=nfolds)
grid_search.fit(X_train, ytarget)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# +
SVM = SVC(kernel='rbf', class_weight='balanced', **grid_result.best_params_)
SVM.fit(X_train_small, y_train_small)

print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(SVM.score(X_train_small, y_train_small)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(SVM.score(X_test_small, y_test_small)))

y_pred_small = SVM.predict(X_test_small)
print("Precision:",metrics.precision_score(y_test_small, y_pred_small))
print("Recall:",metrics.recall_score(y_test_small, y_pred_small))

# + {"endofcell": "--"}
clf = SVM    

kfold = KFold(n_splits=nfolds, random_state=7)
result = cross_val_score(clf, X_train, ytarget, cv=kfold, scoring='accuracy')
print(result.mean())
# -

# Re-evaluate w/ diff. metric due to imbalanced data
result = cross_val_score(clf, X_train, ytarget, cv=kfold, scoring='balanced_accuracy')
print(result.mean())
# --

# +
y_test_pred = clf.predict(X_test_small)
confusion = confusion_matrix(y_test_small, y_test_pred)
print(confusion)

sns.heatmap(confusion, annot=True, cbar=False)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print(classification_report(y_test_small, y_test_pred))
# -

# summarizes the model’s performance by evaluating the trade offs between 
# true positive rate (sensitivity) and false positive rate(1- specificity)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test_small, clf.predict(X_test_small))
fpr, tpr, thresholds = roc_curve(y_test_small, clf.predict_proba(X_test_small)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='SVM Linear (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# run model on the test field
yyy_predict_rbfSVM = clf.predict(XXX)

# ### RFC

# +
from pprint import pprint
rfc = RandomForestClassifier(class_weight='balanced')
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rfc.get_params())

rfc.fit(X_train_small,y_train_small)

print('Accuracy of RF classifier on training set: {:.2f}'
     .format(rfc.score(X_train_small, y_train_small)))
print('Accuracy of RF classifier on test set: {:.2f}'
     .format(rfc.score(X_test_small, y_test_small)))

y_pred_small = rfc.predict(X_test_small)
print("Precision:",metrics.precision_score(y_test_small, y_pred_small))
print("Recall:",metrics.recall_score(y_test_small, y_pred_small))
# -

# ### hyperparam
# - n_estimators = number of trees in the foreset
# - max_features = max number of features considered for splitting a node
# - max_depth = max number of levels in each decision tree
# - min_samples_split = min number of data points placed in a node before the node is split
# - min_samples_leaf = min number of data points allowed in a leaf node
# - bootstrap = method for sampling data points (with or without replacement)

# +
nfolds = 3
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [5, 20, 100]
max_features = ['auto', 'sqrt']
max_depth = [10, 50, 100]
max_depth.append(None)

min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

grid_search = RandomizedSearchCV(estimator=rfc, 
                                 param_distributions=random_grid, 
                                 cv=nfolds,
                                 verbose=2)
grid_search.fit(X_train, ytarget)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# +
rfc = RandomForestRegressor(class_weight='balanced', **grid_result.best_params_)
rfc.fit(X_train_small, y_train_small)

print('Accuracy of rf classifier on training set: {:.2f}'
     .format(rfc.score(X_train_small, y_train_small)))
print('Accuracy of rf classifier on test set: {:.2f}'
     .format(rfc.score(X_test_small, y_test_small)))

y_pred_small = rfc.predict(X_test_small)
print("Precision:",metrics.precision_score(y_test_small, y_pred_small))
print("Recall:",metrics.recall_score(y_test_small, y_pred_small))

# -

from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True], # ......
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rfc = RandomForestRegressor(class_weight='balanced')
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, 
                          cv=nfolds, n_jobs=-1, verbose=2)

# ROC and AUC
logit_roc_auc = roc_auc_score(y_test_small, clf.predict(X_test_small))
fpr, tpr, thresholds = roc_curve(y_test_small, clf.predict_proba(X_test_small)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='RF (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

yyy_predict_rfc = clf.predict(XXX)
