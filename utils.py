import numpy as np

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

