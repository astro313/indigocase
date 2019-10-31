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
from split_fields_to_DF import CreateDFsML, rgb2hsv

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from MLutils import scores, GridSearch_logreg, ROC, confusion_scores, GridSearch_SVMpoly, GridSearch_RFC


def run_ML_pipeline(fname):

    bu = Parser(fname)
    _ = compute_tif_summaryStats(bu.datasetTrain)
    _ = compute_tif_summaryStats(bu.datasetTest)

    # make overlay plot and save to plotField/
    # visualize

    # include NIR
    plot_false_RGB(bu.train_img1, bu.train_img3, bu.train_img4,savedir=bu.plotdir, tag='train', savefig=bu.saveFig)
    plot_false_RGB(bu.test_img1, bu.test_img3, bu.test_img4, savedir=bu.plotdir, tag='test', savefig=bu.saveFig)

    plot_field_all_bands_hist(bu.datasetTrain, bu.saveFig, bu.plotdir, tag='train')
    plot_field_all_bands_hist(bu.datasetTest, bu.saveFig, bu.plotdir, tag='test')

    plot_truthPoints(bu.truth, bu.saveFig, bu.plotdir)
    overplot_truthPoints_trainField(bu.datasetTrain, bu.truth, bu.saveFig, \
                                    bu.plotdir, bu.train_xmin, bu.train_xmax, bu.train_ymin, bu.train_ymax)

    # calc vegetation indices, create pd
    bbb = CreateDFsML(test_size=0.3)
    bbb.outdir = bu.MLplotdir
    bbb.saveFig = bu.saveFig

    bbb.logreg = bu.logreg
    bbb.SVM = bu.SVM
    bbb.RFC = bu.RFC
    bbb.trainndvi = bu.trainndvi
    bbb.trainendvi = bu.trainendvi
    bbb.traincvi = bu.traincvi
    bbb.trainng = bu.trainng
    bbb.trainnnir =  bu.trainnnir
    bbb.trainnr = bu.trainnr
    bbb.traintvi = bu.traintvi

    if bu.trainhue:
        # train also on Hue image
        from split_fields_to_DF import create_rgb
        train_RGB = create_rgb(bu.train_img1, bu.train_img3, bu.train_img4)
        train_hsv_img = rgb2hsv(train_RGB, savedir=bu.plotdir, tag='train',
                               plothist=True, saveFig=bu.saveFig)
        del train_RGB
    else:
        train_hsv_img = None

    bbb.build_DF_trainField(bu.train_img1, bu.train_img2,
                            bu.train_img3, bu.train_img4,
                            bu.truthfile, bu.datasetTrain,
                            train_hsv_img=train_hsv_img,
                            verbose=bu.verbose)
    del train_hsv_img

    bbb.split_data_for_ML()

    # similarly for test
    if bu.trainhue:
        test_RGB = create_rgb(bu.test_img1, bu.test_img3, bu.test_img4)
        test_hsv_img = rgb2hsv(test_RGB, savedir=bu.plotdir, tag='test',
                               plothist=True, saveFig=bu.saveFig)
        del test_RGB
    else:
        test_hsv_img = None

    bbb.build_DF_testField(bu.test_img1, bu.test_img2,
                           bu.test_img3, bu.test_img4,
                           test_hsv_img,
                           bu.verbose)
    del test_hsv_img
    del bu

    # ML pipeline
    if bbb.logreg:
        ## 0 and 1, so start w/ logistic
        logreg = LogisticRegression(penalty='l2', class_weight='balanced')
        logreg.fit(bbb.X_train_small, bbb.y_train_small)
        scores(logreg, 'Logistic Regression', bbb.X_train_small, bbb.y_train_small, bbb.X_test_small, bbb.y_test_small)
        confusion_scores(logreg, bbb.X_test_small, bbb.y_test_small, 'logreg', bbb.outdir, saveFig=bbb.saveFig)
        ROC(logreg, 'LogisticRegression', bbb.X_test_small, bbb.y_test_small,
            bbb.outdir, '', bbb.saveFig)

        # hyperparameter Tuning
        gsc, grid_result = GridSearch_logreg(bbb.X_train_small, bbb.y_train_small
            )
        # apply
        logreg_tuned = LogisticRegression(penalty='l2', class_weight='balanced', **grid_result.best_params_)
        logreg_tuned.fit(bbb.X_train_small, bbb.y_train_small)
        scores(logreg_tuned, 'Logistic Regression, Tuned', bbb.X_train_small, bbb.y_train_small, bbb.X_test_small, bbb.y_test_small)
        confusion_scores(logreg_tuned, bbb.X_test_small, bbb.y_test_small, 'logregtuned', bbb.outdir, saveFig=bbb.saveFig)
        ROC(logreg_tuned, 'LogisticRegressionTuned', bbb.X_test_small, bbb.y_test_small, bbb.outdir, '', bbb.saveFig)

        # run model on the test field
        yyy_predict_logreg = logreg_tuned.predict(bbb.XXX)
        print("number of plants in test field: ", yyy_predict_logreg.sum())
        print("{:.2f}% of field", (yyy_predict_logreg.sum()/len(yyy_predict_logreg))*100)


    if bbb.SVM:
        # SVM
        from sklearn.svm import SVC

        SVM = SVC(kernel='linear', class_weight='balanced')
        SVM.fit(bbb.X_train_small, bbb.y_train_small)
        scores(SVM, 'SVM linear', bbb.X_train_small, bbb.y_train_small, bbb.X_test_small, bbb.y_test_small)
        confusion_scores(SVM, bbb.X_test_small, bbb.y_test_small, 'SVM linear', bbb.outdir, saveFig=bbb.saveFig)

        # SVM poly
        SVM = SVC(kernel='poly', class_weight='balanced')
        SVM.fit(bbb.X_train_small, bbb.y_train_small)
        scores(SVM, 'SVM poly', bbb.X_train_small, bbb.y_train_small, bbb.X_test_small, bbb.y_test_small)
        confusion_scores(SVM, bbb.X_test_small, bbb.y_test_small, 'SVM poly', bbb.outdir, saveFig=bbb.saveFig)

        # hypertuning for poly
        gsc, grid_result =  GridSearch_SVMpoly(bbb.X_train_small, bbb.y_train_small
            )
        # apply
        SVMpoly_tuned = SVC(kernel='poly', class_weight='balanced', **grid_result.best_params_)
        SVMpoly_tuned.fit(bbb.X_train_small, bbb.y_train_small)
        scores(SVMpoly_tuned, 'SVM poly, Tuned', bbb.X_train_small, bbb.y_train_small, bbb.X_test_small, bbb.y_test_small)
        confusion_scores(SVMpoly_tuned, bbb.X_test_small, bbb.y_test_small, 'SVMpolytuned', bbb.outdir, saveFig=bbb.saveFig)
        ROC(SVMpoly_tuned, 'SVMpolyTuned', bbb.X_test_small, bbb.y_test_small,
            bbb.outdir, '', bbb.saveFig)

        # run model on the test field
        yyy_predict_poly = SVMpoly_tuned.predict(bbb.XXX)
        print("number of plants in test field: ", yyy_predict_poly.sum())
        print("{:.2f}% of field", (yyy_predict_poly.sum()/len(yyy_predict_poly))*100)


        # SVM rbf
        SVM = SVC(kernel='rbf', class_weight='balanced')
        SVM.fit(bbb.X_train_small, bbb.y_train_small)
        scores(SVM, 'SVM rbf', bbb.X_train_small, bbb.y_train_small, bbb.X_test_small, bbb.y_test_small)
        confusion_scores(SVM, bbb.X_test_small, bbb.y_test_small, 'SVM rbf', bbb.outdir, saveFig=bbb.saveFig)

        # hypertuning for rbf
        gsc, grid_result =  GridSearch_SVMrbf(bbb.X_train_small, bbb.y_train_small
            )
        # apply
        SVMrbf_tuned = SVC(kernel='rbf', class_weight='balanced', **grid_result.best_params_)
        SVMrbf_tuned.fit(bbb.X_train_small, bbb.y_train_small)
        scores(SVMrbf_tuned, 'SVM rbf, Tuned', bbb.X_train_small, bbb.y_train_small, bbb.X_test_small, bbb.y_test_small)
        confusion_scores(SVMrbf_tuned, bbb.X_test_small, bbb.y_test_small, 'SVMrbftuned', bbb.outdir, saveFig=bbb.saveFig)
        ROC(SVMrbf_tuned, 'SVMrbfTuned', bbb.X_test_small, bbb.y_test_small,
            bbb.outdir, '', bbb.saveFig)

        # run model on the test field
        yyy_predict_rbf = SVMrbf_tuned.predict(bbb.XXX)
        print("number of plants in test field: ", yyy_predict_rbf.sum())
        print("{:.2f}% of field", (yyy_predict_rbf.sum()/len(yyy_predict_rbf))*100)


    if bbb.RFC:
        ### RFC
        rfc = RandomForestClassifier(class_weight='balanced')
        rfc.fit(X_train_small,y_train_small)

        # Look at parameters used by our current forest
        from pprint import pprint
        print('Parameters currently in use:\n')
        pprint(rfc.get_params())

        scores(rfc, 'RFC', bbb.X_train_small, bbb.y_train_small, bbb.X_test_small, bbb.y_test_small)
        confusion_scores(rfc, bbb.X_test_small, bbb.y_test_small, 'RFC', bbb.outdir, saveFig=bbb.saveFig)

        # hypertuning for RFC
        gsc, grid_result =  GridSearch_RFC(bbb.X_train_small, bbb.y_train_small
            )
        # apply
        rfc_tuned = RandomForestRegressor(class_weight='balanced', **grid_result.best_params_)
        rfc_tuned.fit(bbb.X_train_small, bbb.y_train_small)
        scores(rfc_tuned, 'RFC Tuned', bbb.X_train_small, bbb.y_train_small, bbb.X_test_small, bbb.y_test_small)
        confusion_scores(rfc_tuned, bbb.X_test_small, bbb.y_test_small, 'RFCtuned', bbb.outdir, saveFig=bbb.saveFig)
        ROC(rfc_tuned, 'RFCTuned', bbb.X_test_small, bbb.y_test_small,
            bbb.outdir, '', bbb.saveFig)

        # run model on the test field
        yyy_predict_rfc = rfc_tuned.predict(bbb.XXX)
        print("number of plants in test field: ", yyy_predict_rfc.sum())
        print("{:.2f}% of field", (yyy_predict_rfc.sum()/len(yyy_predict_rfc))*100)



if __name__ == "__main__":

    try:
        fname = sys.argv[1]
    except:
        fname = 'config.yaml'

    run_ML_pipeline(fname)



