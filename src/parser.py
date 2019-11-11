import matplotlib
new_style = {'grid': False}
matplotlib.rc('axes', **new_style)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import shape
# from osgeo import gdal, osr
import os


class Parser(object):

    def __init__(self, yfname):
        self.yfname = yfname
        self.parse_yaml_file()
        self.read_train_data()
        self.read_test_data()
        self.read_truth_data()

    def parse_yaml_file(self):
        import config
        ccc = config.Configurable(self.yfname)
        setupParam = ccc.config_dict

        dpath = setupParam['input']['dataPath']
        self.trainfile = os.path.join(dpath, setupParam['input']['trainFile'])
        self.testfile = os.path.join(dpath, setupParam['input']['testFile'])
        self.truthfile = os.path.join(dpath, setupParam['input']['truthFile'])

        assert os.path.isfile(self.trainfile)
        assert os.path.isfile(self.testfile)
        assert os.path.isfile(self.truthfile)

        self.plotdir = setupParam['output']['plotdir']
        self.MLplotdir = setupParam['output']['MLplotdir']

        if not os.path.exists(self.plotdir):
            os.makedirs(self.plotdir)

        if not os.path.exists(self.MLplotdir):
            os.makedirs(self.MLplotdir)

        self.logreg = setupParam['ML']['logreg']
        self.SVM = setupParam['ML']['SVM']
        self.RFC = setupParam['ML']['RFC']

        self.saveFig = setupParam['misc']['saveFig']
        self.verbose = setupParam['misc']['verbose']

        self.trainhue = setupParam['ML']['trainhue']
        self.trainndvi = setupParam['ML']['trainndvi']
        self.trainendvi = setupParam['ML']['trainendvi']
        self.traincvi = setupParam['ML']['traincvi']
        self.trainng = setupParam['ML']['trainng']
        self.trainnnir = setupParam['ML']['trainnnir']
        self.trainnr = setupParam['ML']['trainnr']
        self.traintvi = setupParam['ML']['traintvi']


    def read_train_data(self):
        self.datasetTrain = rasterio.open(self.trainfile)

        if self.verbose:
            print(self.datasetTrain.meta)
            print("No data values for all channels: ", self.datasetTrain.nodatavals)

        with rasterio.open(self.trainfile) as src:
            profile = src.profile
            (self.train_xmin, self.train_ymin, self.train_xmax, self.train_ymax) = src.bounds

            if self.verbose:
                print(src.profile)

                # width = src.width
                # height = src.height
            #    print(src.descriptions)

            self.train_img1 = src.read(1).astype(float)
            self.train_img2 = src.read(2).astype(float)
            self.train_img3 = src.read(3).astype(float)
            self.train_img4 = src.read(4).astype(float)



    def read_test_data(self):
        from utils import normalize

        self.datasetTest = rasterio.open(self.testfile)

        if self.verbose:
            print(self.datasetTest.meta)
            print("No data values for all channels: ", self.datasetTest.nodatavals)

        with rasterio.open(self.testfile) as src:
            profile = src.profile
            if self.verbose:
                print(src.profile)

                # (xmin, ymin, xmax, ymax) = src.bounds
                # width = src.width
                # height = src.height
            #    print(src.descriptions)

            self.test_img1 = src.read(1).astype(float)
            self.test_img2 = src.read(2).astype(float)
            self.test_img3 = src.read(3).astype(float)
            self.test_img4 = src.read(4).astype(float)


    def read_truth_data(self):
        self.truth = gpd.read_file(self.truthfile)

        if self.verbose:
            print("Cols in Truth: ", self.truth.columns)
            print("Num of Plants in Truth: ", len(self.truth))
            print(self.truth.geometry[:5])
            print("CRS of Truth data: ", self.truth.crs)

