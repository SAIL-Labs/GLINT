import sys
sys.path.append('/home/scexao/glint/control-code/')
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import shmDMcontrol
from pyMilk.interfacing.shm import SHM
import time


# Tilt all segments away except one, and scan that one through piston between -1.3 to 1.3 dm microns. Then save the glintpg1 frame to the folder psfimgs

dm = shmDMcontrol.DM()
psf = SHM('glintpg2')
scan_seg_num = 31

def tilt_all_except(segment):
    for i in range(37):
        if i != segment:
            dm.set_segment(i, 0, 0, 6)
    return

dmpos = np.arange(-1.3, 1.3, 0.1)

tilt_all_except(scan_seg_num)
imgs = []

for pos in dmpos:
    
    dm.set_segment(scan_seg_num, pos, 0, 0)
    time.sleep(0.5)

    # Save the PSF image

    psf_img = psf.get_data()
    opd = pos*2

    imgs.append(psf_img)


# Save the images to a FITS file
fits_path = f'psfimgs4/scanseg{scan_seg_num}.fits'


opds = np.array(dmpos) * 2  # Convert to microns

# Primary HDU: image stack
primary_hdu = fits.PrimaryHDU(np.array(imgs))

# Secondary HDU: OPDs
opd_hdu = fits.ImageHDU(opds, name='OPDS')

# Combine into HDU list
hdul = fits.HDUList([primary_hdu, opd_hdu])

# Write to FITS file
hdul.writeto(fits_path, overwrite=True)