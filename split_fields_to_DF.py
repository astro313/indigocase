"""

    calc ndvi, create pd, split field into trainning and test set for ML model

"""

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from rasterio.plot import show
from shapely.geometry import shape
import os
from parser import Parser
import matplotlib.pyplot as plt


def create_rgb(r, g, b, stretch=0.5):
    from astropy.visualization import make_lupton_rgb
    image = make_lupton_rgb(r, g, b, stretch=stretch)
    return image


def rgb2hsv(rgbim, savedir='./', tag='', plothist=False, saveFig=False):
    from skimage.color import rgb2hsv
    hsv_img = rgb2hsv(rgbim)

    if plothist:
        plothist_hsv(hsv_img, savedir, saveFig, tag)
    return hsv_img


def plothist_hsv(hsv_img, savedir, saveFig, tag):
    hue_img = hsv_img[:, :, 0]
    value_img = hsv_img[:, :, 2]

    fig, ax = plt.subplots()

    ax.hist(hue_img.ravel(), 512)
    ax.set_title("Histogram of the Hue channel")

    if saveFig:
        if tag != '':
            tag += '_'
        plt.savefig(savedir + tag + 'hue_channel.png', bbox_inches='tight')
    else:
        plt.show()


class CreateDFsML(object):
    def __init__(self, test_size):
        self.test_size=0.3
        self.saveFig = False
        self.outdir = './'


    def ndviCalc(self, red, nir, verbose):
        """

        Parameters
        ----------

        red: numpy.ndarray
            2d image

        nir: numpy.ndarray
            2d image

        plot: bool

        verbose: bool

        Returns
        -------
        ndvi: ndarray
            ndvi values in 2D image

        badii: bool array
            index in array where ndvi is np.nan

        """
        ndvi = ((nir - red)/(nir + red))

        beforeNnans = np.isnan(ndvi).sum()
        ndvi[np.where(np.logical_or(ndvi < 0, ndvi > 1)) ] = np.nan

        if verbose:
            print(beforeNnans, np.isnan(ndvi).sum())

            print("mean ndvi: ", np.nanmean(ndvi))
            print("std ndvi", np.nanstd(ndvi))

        # remove rows where ndvi is nan, since ML can train on those rows
        badii = np.isnan(ndvi)

        if verbose:
            print("Num of pixels with nan in ndvi: ", badii.sum())
            print("Total number of pixels: ", len(ndvi.flatten()))

        return ndvi, badii


    def build_DF_trainField(self, train_img1, train_img2,
                                  train_img3, train_img4,
                            truthf, datasetTrain,
                            train_hsv_img=None,
                            verbose=False):

        ndvi, badii = self.ndviCalc(train_img1, train_img4, verbose)
        # build ytarget based on ground truth
        ytarget = np.zeros_like(train_img1)

        import fiona
        with fiona.open(truthf, "r") as shfile:
            features = [feature['geometry'] for feature in shfile]

        for i in features:
            # get the x,y pix position of each plant pos
            py, px = datasetTrain.index(*i['coordinates'])
            # print('Pixel Y, X coords: {}, {}'.format(py, px))
            ytarget[py, px] = 1.0
        self.ytarget = ytarget[~badii].flatten().astype(int)

        # build DF as X
        X_train = pd.DataFrame()
        X_train['b1'] = train_img1[~badii].flatten()
        X_train['b2'] = train_img2[~badii].flatten()
        X_train['b3'] = train_img3[~badii].flatten()
        X_train['b4'] = train_img4[~badii].flatten()
        X_train['ndvi'] = ndvi[~badii].flatten()

        if train_hsv_img is not None:
            X_train['hueIm'] = train_hsv_img[:, :, 0][~badii].flatten()
            X_train['valueIm'] = train_hsv_img[:, :, 2][~badii].flatten()

        cnames = X_train.columns.values

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        self.X_train = pd.DataFrame(X_train, columns=cnames)
        if verbose:
            self.X_train.describe()

        assert self.ytarget.shape[0] == self.X_train.shape[0]


    def split_data_for_ML(self):

        """
        ytarget: pd.DataFrames
            1 if pixel or row is a plant; else 0

        X_train: pd.DataFrames
            readout of each band per pix --> row

        """
        from sklearn.model_selection import train_test_split
        self.X_train_small, self.X_test_small, self.y_train_small, self.y_test_small = train_test_split(self.X_train, self.ytarget, test_size=self.test_size)


    def build_DF_testField(self, test_img1, test_img2, test_img3, test_img4,
                           test_hsv_img=None,
                           verbose=False):

        ndviTest, badiiTest = self.ndviCalc(test_img1, test_img4, verbose)
        # build DF as X
        XXX = pd.DataFrame()
        XXX['b1'] = test_img1[~badiiTest].flatten()
        XXX['b2'] = test_img2[~badiiTest].flatten()
        XXX['b3'] = test_img3[~badiiTest].flatten()
        XXX['b4'] = test_img4[~badiiTest].flatten()
        XXX['ndvi'] = ndviTest[~badiiTest].flatten()

        if test_hsv_img is not None:
            XXX['hueIm'] = test_hsv_img[:, :, 0][~badiiTest].flatten()
            XXX['valueIm'] = test_hsv_img[:, :, 2][~badiiTest].flatten()

        cnames = XXX.columns.values

        # generate X for the test field
        scaler = MinMaxScaler()
        XXX = scaler.fit_transform(XXX)
        self.XXX = pd.DataFrame(XXX, columns=cnames)

        if verbose:
            self.XXX.describe()

