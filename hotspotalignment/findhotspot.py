

import numpy as np
import centroid
from astropy.io import fits
import matplotlib.pyplot as plt


def crop(frames, y_start=0, y_end=-1, x_start=0, x_end=-1):
        """
        Crop frames to zoom in on the PSF for better centroid fitting.
        
        Parameters:
        frames (ndarray): Input frames.
        y_start, y_end, x_start, x_end (int): Coordinates for cropping.
        
        Returns:
        ndarray: Cropped frames.
        """

        return frames[y_start:y_end, x_start:x_end].astype(float)

def find_origin_com(img, threshold):
    """
    Find the center of mass (COM) of the image based on a threshold.
    
    Parameters:
    img (ndarray): Input image.
    threshold (float): Threshold factor (0 to 1). 
    
    Returns:
    tuple: Coordinates of the COM (x, y).
    """
    max_val = np.nanmax(img)
    threshold  = threshold*max_val
    return centroid.centroid(img, threshold)


'''
Below is an example use for the two functions above.
'''

if __name__ == "__main__":

    # Get the data
    dark = fits.getdata('frames/summitalignment_03_03_25/psf_dark.fits').astype(float)
    bright = fits.getdata('frames/summitalignment_03_03_25/psf_zerovolts_ir2.fits').astype(float)
    img = bright - dark

    # Crop the portion of the hotspot for better estimation
    img = crop(img, y_start=200, y_end=800, x_start=200, x_end=800)

    # Set an intensity threshold so that only pixels bright than this will be considered for the COM calculation.
    threshold = 0.9

    # Find the COM
    xstart, ystart = find_origin_com(img, threshold=threshold)
    print(f'Center of mass: {xstart}, {ystart}')

    # Plot the cropped image
    plt.imshow(img, cmap='gray')
    plt.plot(xstart, ystart, 'rx', markersize=4)
    plt.show()



