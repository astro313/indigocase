import matplotlib
new_style = {'grid': False}
matplotlib.rc('axes', **new_style)
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import shape
# from osgeo import gdal, osr

import os, sys
import config
import yaml
from parser import Parser
from utils import compute_tif_summaryStats
from visutil import plot_false_RGB, plot_field_all_bands_hist, plot_truthPoints, overplot_truthPoints_trainField
from split_fields_to_DF import CreateDFsML

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def run(fname):

    bu = Parser(fname)
    _ = compute_tif_summaryStats(bu.datasetTrain)
    _ = compute_tif_summaryStats(bu.datasetTest)

    # make overlay plot and save to plotField/
    # visualize
    plot_false_RGB(bu.train_img1, bu.train_img2, bu.train_img3, savedir=bu.plotdir, tag='train', savefig=bu.saveFig)
    plot_false_RGB(bu.test_img1, bu.test_img2, bu.test_img3, savedir=bu.plotdir, tag='test', savefig=bu.saveFig)

    plot_field_all_bands_hist(bu.datasetTrain, bu.saveFig, bu.plotdir, tag='train')
    plot_field_all_bands_hist(bu.datasetTest, bu.saveFig, bu.plotdir, tag='test')

    plot_truthPoints(bu.truth, bu.saveFig, bu.plotdir)
    overplot_truthPoints_trainField(bu.datasetTrain, bu.truth, bu.saveFig, \
                                    bu.plotdir, bu.train_xmin, bu.train_xmax, bu.train_ymin, bu.train_ymax)

    # calc ndvi, create pd
    bbb = CreateDFsML(test_size=0.3)
    bbb.outdir = bu.MLplotdir
    bbb.saveFig = bu.saveFig
    bbb.build_DF_trainField(bu.train_img1, bu.train_img2, bu.train_img3, bu.train_img4, bu.truthfile, bu.datasetTrain, bu.verbose)
    bbb.split_data_for_ML()
    bbb.build_DF_testField(bu.test_img1, bu.test_img2, bu.test_img3, bu.test_img4, bu.verbose)
    del bu

    # ML pipeline
    ## 0 and 1, so start w/ logistic
    logreg = LogisticRegression(penalty='l2', class_weight='balanced')
    logreg.fit(bbb.X_train_small, bbb.y_train_small)
    scores(logreg, 'Logistic Regression', bbb.X_train_small, bbb.y_train_small, bbb.X_test_small, bbb.y_test_small)

    # hyperparameter Tuning
    from MLutils import GridSearch_logreg
    gsc, grid_result = GridSearch_logreg(bbb.X_train_small, bbb.y_train_small
        )
    # apply
    logreg_tuned = LogisticRegression(penalty='l2', class_weight='balanced', **grid_result.best_params_)
    logreg_tuned.fit(bbb.X_train_small, bbb.y_train_small)
    scores(logreg_tuned, 'Logistic Regression, Tuned', bbb.X_train_small, bbb.y_train_small, bbb.X_test_small, bbb.y_test_small)
    confusion_scores(logreg_tuned, bbb.X_test_small, bbb.y_test_small, 'logregtuned', bbb.outdir, saveFig=bbb.saveFig)

    # run model on the test field
    clf = logreg       # logreg_tuned
    yyy_predict_logreg = clf.predict(bbb.XXX)
    print("number of plants in test field: ", yyy_predict_logreg.sum())
    print("{:.2f}% of field", (yyy_predict_logreg.sum()/len(yyy_predict_logreg))*100)
    ROC('LogisticRegressionTuned', bbb.X_test_small, bbb.y_test_small,
        bbb.outdir, bbb.saveFig)



if __name__ == "__main__":

    # fname = 'config.yaml'
    fname = sys.argv[1]
    run(fname)



