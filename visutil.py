import numpy as np
import matplotlib.pyplot as plt
import os
from utils import normalize

def plot_false_RGB(im1, im2, im3, savedir, gamma=0.5, perlow=5, perhigh=98, tag='', savefig=True):
    """
    Make false color RGB from 3 bands of images and plot

    Parameters
    ----------
    im1: numpy array
        b1 image, 2d

    im2: numpy array
        b2 image, 2d

    im3: 2d array
        b3 image, 3d

    gamma: float
        gamma correction value

    perlow: int or float
        lower bound for percentile for imshow constrast stretch

    perhigh: int or float
        uppper bound for percentile for imshow constrast stretch

    savedir: str
        where to save plots

    tag: str
        e.g., 'train' or 'test'

    savefig: bool
        save plot or show


    """

    # Normalize band
    im3_norm = normalize(im3)
    im2_norm = normalize(im2)
    im1_norm = normalize(im1)

    false_RGB = np.dstack((im1_norm, im2_norm, im3_norm))

    plt.figure()
    plt.hist(false_RGB.flatten())
    if savefig:
        plt.savefig(savedir + tag + '_falsergb_hist.png')
    else:
        plt.show()
    plt.close()

    # scale up contrast
    from skimage import exposure
    plow, phigh = np.percentile(false_RGB, (perlow, perhigh))
    rgbStretched = exposure.rescale_intensity(false_RGB, in_range=(plow, phigh)) # Perform contrast stretch on RGB range
    rgbStretched = exposure.adjust_gamma(rgbStretched, gamma)                  # Perform Gamma Correction

    # imshow
    fig = plt.figure()
    ax = plt.Axes(fig,[0,0,1,1])
    ax.set_axis_off()
    fig.add_axes(ax)
    # Plot a natural color RGB
    ax.imshow(rgbStretched, interpolation='bilinear', alpha=0.9)
    if savefig:
        plt.savefig(savedir + tag + '_falsergb.png')
    else:
        plt.show()
    plt.close()


def plot_field(data, savefig, plotdir, tag='train'):

    """
    plot field

    Parameters
    ----------
    data: DatasetReader
        data read in from rasterio
    tag: str
        e.g., 'train' or 'test'

    """



    show(data)
    if savefig:
        plt.savefig(os.path.join(plotdir, tag + '_basic.png'),
                    bbox_inches="tight")
    else:
        plt.show()


def plot_field_all_bands_hist(data, savefig, plotdir, tag=''):
    """
    plot histogram of all 4 bands in the tif data.

    Parameters
    ----------
    data: DatasetReader
        data read in from rasterio

    savefig: bool
        savefig or show

    plotdir: str
        path to savefig

    tag: str
        e.g., 'train' or 'test'

    """

    plt.figure()
    data = data.read()
    plt.hist(data.flatten(), bins=50, lw=0.0, stacked=False, alpha=0.3,
            histtype='stepfilled')
    plt.title(tag)

    if savefig:
        plt.savefig(os.path.join(plotdir, tag + '_hist.png'),
                    bbox_inches="tight")
    else:
        plt.show()



def plot_truthPoints(truth, saveFig, plotdir, tag=''):
    """
    plot truth "Points" geometry

    TODO: check CRS of truth and train field before overlaying

    """
    fig, ax = plt.subplots(figsize = (10,10))
    truth.plot(ax=ax)
    ylim = plt.gca().get_ylim()
    xlim = plt.gca().get_xlim()
    if saveFig:
        tag += '_'
        plt.savefig(os.path.join(plotdir, tag + 'truth.png'),
                    bbox_inches="tight")
    else:
        plt.show()


def overplot_truthPoints_trainField(datasetTrain, truth, savefig, plotdir, xmin=None, xmax=None, ymin=None, ymax=None, tag=''):

    """
    datasetTrain: DatasetReader
        containing the 4 bands images from the training field

    truth: geopandas
        containing the "Point" geometry of the truth in the same field

    """
    from rasterio.plot import show
    fig, ax = plt.subplots()
    show((datasetTrain, 1), ax=ax)
    truth.plot(ax=ax, color='red', alpha=0.05)

    plt.title('Truth overplotted on Train field')

    if savefig:
        tag += '_'
        plt.savefig(os.path.join(plotdir, tag + 'truthONtrain.png'),
                    bbox_inches="tight")
    else:
        plt.show()

