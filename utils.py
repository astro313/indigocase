import numpy as np
from math import *

def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)


def compute_tif_summaryStats(data):
    """

    data: DatasetReader
        data read in from rasterio

    """

    # read all bands
    array = data.read()

    stats = []
    for band in array:
         stats.append({
             'min': band.min(),
             'mean': band.mean(),
             'median': np.median(band),
             'max': band.max()})
    print(stats)
    return stats


def get_nrows_ncols_from_tif(tifname, verbose=True):
    """

    """
    from osgeo import gdal, osr
    ds = gdal.Open(tifname)

    if verbose:
        print(ds.GetProjection())

    cols = ds.RasterXSize; print('# of columns:',cols)
    rows = ds.RasterYSize; print('# of rows:',rows)

    return cols, rows


def get_ellps_from_tif(ds):
    """
    Get ellps Meta data from TIF
    """
    ppp = ds.GetProjection()
    starti = ppp.find('GEOGCS')
    _end = ppp[starti+8:]
    end = _end.find('"')
    # print(end)
    bubub = _end[:end].replace(" ", "")
    return bubub


def pixel_unit_deg_from_tif(ds):
    # check if pixel unit is degree
    ppp = ds.GetProjection()
    starti = ppp.find('UNIT')
    if ppp[starti+6: starti+12] == 'degree':
        print("Map in real world units...")
        return True
    else:
        return False


def get_geo_from_tif(tifname, verbose=True):
    from osgeo import gdal, osr
    ds = gdal.Open(tifname)

    geo_transform = ds.GetGeoTransform()
    # https://download.osgeo.org/gdal/workshop/foss4ge2015/workshop_gdal.pdf

    # left-most X coord, W-E pixel res, rotation (0 if N-up),
    # upper Y coor, rotation, N-S pix resolution
    xMin, dx, b, yMax, d, dy = geo_transform

    if verbose:
        if dy < 0:
            print("origin of image is upper left corner")

        if pixel_unit_deg_from_tif(ds):
            print("pixel size in x-dir, y-dir [deg]: ", dx, dy)    # in deg
        else:
            print("pixel size in x-dir, y-dir [some units..]: ", dx, dy)

        print("xMin, yMax: ", xMin, yMax)

    return geo_transform


def pixel2coord(geo_transform, x, y):
    """
    Returns global coordinates from pixel x, y coords
    """
    xoff, a, b, yoff, d, e = geo_transform

    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff
    return (xp, yp)



def get_latlon_axeslab(geo_transform, x, y):
    lat, lon = pixel2coord(geo_transform, x, y)
    return lat, lon0


def get_raster_extents_from_tif(gt, ds):
    width = ds.RasterXSize
    height = ds.RasterYSize

    print('Upper Left Corner:', gdal.ApplyGeoTransform(gt, 0, 0))
    print('Upper Right Corner:', gdal.ApplyGeoTransform(gt, width, 0))
    print('Lower Left Corner:', gdal.ApplyGeoTransform(gt,0, height))
    print('Lower Right Corner:',gdal.ApplyGeoTransform(gt, width, height))
    print('Center:', gdal.ApplyGeoTransform(gt, width/2, height/2))

    # similarly,
    minx = gt[0]
    miny = gt[3] + width*gt[4] + height*gt[5]
    maxx = gt[0] + width*gt[1] + height*gt[2]
    maxy = gt[3]
    print(minx, maxx, miny, maxy)
    return minx, maxx, miny, maxy


def dist_btw_pair_latlon(lon0, lat0, lon1, lat1):
    d2r =0.0174532925199433
    dlon = (lon1 - lon0)*d2r
    dlat = (lat1 - lat0)*d2r

    a = (sin(dlat/2))**2 + cos(lat0) * cos(lat1) * (sin(dlon/2))**2
    c = 2 * atan2( sqrt(a), sqrt(1-a) )
    RE = 6378.1e3     # m
    d = RE * c
    return d


