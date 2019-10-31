"""

Make plant population map.


"""

import numpy as np
import matplotlib.pyplot as plt


def compute_plperac(ds, plant_bool, verbose=False):
    """
        determine the number of plants/acre from 2D bool array

    Parameters
    ---------
    plant_bool: array of bools
        1 meaning it contains a plant, 0 else

    Returns
    -------
    new: array
        containing number of plants per acre, which is the area of each pixel

    """
    from utils import get_m_per_px, get_raster_extents, dist_btw_pair_latlon

    minx, maxx, miny, maxy = get_raster_extents(geo_transform)

    # get the extent of the train field in physical units
    yyyrange_m = dist_btw_pair_latlon(minx, miny,  minx, maxy)
    xxxrange_m = dist_btw_pair_latlon(minx, miny,  maxx, miny)

    if verbose:
        print("length spanned by map: {:.2f} [m] x {:.2f} [m]".format(xxxrange_m, yyyrange_m))

    ac2sqm = 4046.86
    p = np.sqrt(ac2sqm)
    delx = p
    dely = p

    # assuming in meters
    xmin, xmax, ymin, ymax = 0, xxxrange_m, 0, yyyrange_m

    ny = int(np.ceil(np.abs(ymax - ymin)/p))
    nx = int(np.ceil(np.abs(xmax - xmin)/p))
    new = np.zeros((ny, nx))
    # print(ny, nx)

    # number of elements in original image w/in p [meters]
    cols = plant_bool.shape[1]
    rows = plant_bool.shape[0]

    mPerPixX, mPerPixY = xxxrange_m / cols, yyyrange_m / rows
    nnx = int(p / mPerPixX)
    nny = int(p / mPerPixY)

    xticklab = []
    yticklab = []
    for k in range(nx):
        for l in range(ny):
            new[l, k] = plant_bool[nny*l: nny*(l+1), nnx*k: nnx*(k+1)].sum()

    return new


def plot_plant_pop_map(plant_pop_arr, coord=None, plotdir='./', saveFig=False):
    """
    visualize # of plants/acer

    """
    if plant_pop_arr.ndim == 2:
        fig, ax = plt.subplots()
        plt.imshow(plant_pop_arr)
        plt.colorbar()

        if coord is not None:
            raise NotImplementedError
        plt.title('Plant population map [# plants/ac]')
        if saveFig:
            plt.savefig(plotdir + 'plant_population_map.png', bbox_inches='tight')
        else:
            plt.show()


if __name__ == '__main__':
    from split_fields_to_DF import CreateDFsML
    from parser import Parser
    from utils import get_raster_extents_from_tif, dist_btw_pair_latlon
    from osgeo import gdal

    try:
        fname = sys.argv[1]
    except:
        fname = 'config.yaml'


    bu = Parser(fname)
    bbb = CreateDFsML()
    plant_bool = bbb.bool_mask_truth(bu.datasetTrain, bu.truthfile)

    ds = gdal.Open(bu.trainfile)
    geo_transform = ds.GetGeoTransform()
    plantcount_map = compute_plperac(ds, plant_bool, verbose=bu.verbose)
    plot_plant_pop_map(plantcount_map, plotdir=bu.outdir, saveFig=bu.saveFig)





