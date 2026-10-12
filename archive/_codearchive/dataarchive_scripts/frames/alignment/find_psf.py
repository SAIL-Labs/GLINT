import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
import glint_paths


def centroid(image, threshold=0):
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

    signal = np.where(image > threshold)
    sy, sx = image.shape[0], image.shape[1]

    temp = np.zeros((sy, sx))
    temp[signal] = image[signal]

    profx = 1.0 * temp.sum(axis=0)
    profy = 1.0 * temp.sum(axis=1)
    profx -= np.min(profx)
    profy -= np.min(profy)

    x0 = (profx*np.arange(sx)).sum() / profx.sum()
    y0 = (profy*np.arange(sy)).sum() / profy.sum()

    return (x0, y0)

def find_origin_com(img, threshold = 0.6):
    # Find the origin of the image

    # find max value in img
    max_val = np.max(img)
    # threshold the image
    threshold  = threshold*max_val

    (x,y) = centroid(img, threshold)

    return x,y

if __name__ == '__main__':

    # OG alignment at the summit
    goalpath = str(glint_paths.DATA_ROOT / '_dataarchive' / 'frames' / 'alignment' / 'summitalignment') + '/'
    goalfilename = "glint_focal_2024-07-12_23:57:23.706502.fits"
    goaldarkfilename = 'glint_focal_2024-07-12_20:45:39.688003.fits'

    hdul = fits.open(goalpath + goalfilename)
    goalimg = hdul[0].data
    goalimg = goalimg.astype(float)

    hdul = fits.open(goalpath + goaldarkfilename)
    goaldark = hdul[0].data
    goaldark = goaldark.astype(float)

    goalimg = goalimg - goaldark

    # normalise
    goalimg = goalimg/np.max(goalimg)

    # compare current alignment to the original alignment at the summit
    path = str(glint_paths.data_dir('_dataarchive', 'frames', 'alignment')) + '/'
    if len(sys.argv) == 1:
        print('Please provide the filename of the current image')
        sys.exit(1)
    elif len(sys.argv) > 2:
        print('Please provide only one filename')
        sys.exit(1)
    filename = sys.argv[1] 
    darkfilename = 'dark_psf.fits'

    hdul = fits.open(path + filename)
    img = hdul[0].data
    img = img.astype(float)

    hdul = fits.open(path + darkfilename)
    dark = hdul[0].data
    dark = dark.astype(float)

    img = img - dark

    # normalise
    img = img/np.max(img)

    # Crop the image
    start_x = 440
    start_y = 360
    end_x = 570
    end_y = 490
    # start_x = 0
    # start_y = 0
    # end_x = -1
    # end_y = -1

    goalimg = goalimg[start_y:end_y,start_x:end_x]
    img = img[start_y:end_y,start_x:end_x]

    # Find origin of the image
    goalx,goaly = find_origin_com(goalimg)
    print('goalx: {:.1f}, goaly: {:.1f}'.format(goalx,goaly))

    x,y = find_origin_com(img)
    print('x: {:.1f}, y: {:.1f}'.format(x,y))

    offset = np.sqrt((goalx-x)**2 + (goaly-y)**2)


    # Plot the images as a subplot
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(121)
    plt.imshow(goalimg, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.plot(goalx, goaly, 'rx')
    plt.title('Goal (12/07/24) PSF centre: ({:.1f}, {:.1f})'.format(goalx,goaly))

    ax = fig.add_subplot(122)
    plt.imshow(img, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.plot(x, y, 'rx')
    plt.title('Current PSF centre: ({:.1f}, {:.1f})\n offset: {:.1f}'.format(x,y, offset))

    # plt.savefig('alignmentcheck__08_2024.png')

    plt.show()
