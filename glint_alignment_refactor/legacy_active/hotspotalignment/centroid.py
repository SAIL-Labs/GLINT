import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt



def centroid(image, threshold):
    ''' ------------------------------------------------------
    Determines the center of gravity of an array

    Parameters:
    ----------
    - image: the array
    - threshold: value above which pixels are taken into account
    - binarize: binarizes the image before centroid (boolean)

    Remarks:
    -------
    The binarize option can be useful for apertures, expected
    to be uniformly lit.
    ------------------------------------------------------ '''

    # Get only the bright pixels
    signal = np.where(image > threshold)
    sy, sx = image.shape[0], image.shape[1]

    # Get a threhsold array
    temp = np.zeros((sy, sx))
    temp[signal] = image[signal]

    # Intensity profiles along x and y axes
    profx = 1.0 * temp.sum(axis=0)
    profy = 1.0 * temp.sum(axis=1)

    # Shift the minimum to zero
    profx -= np.min(profx)
    profy -= np.min(profy)

    # # Smooth the profiles to reduce noise
    profx = gaussian_filter(profx, sigma=2)
    profy = gaussian_filter(profy, sigma=2)

    # Centroid calculation
    x0 = (profx*np.arange(sx)).sum() / profx.sum()
    y0 = (profy*np.arange(sy)).sum() / profy.sum()

    return (x0, y0)