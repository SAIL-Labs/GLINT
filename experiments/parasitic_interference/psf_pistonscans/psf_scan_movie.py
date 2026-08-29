import numpy as np  
import matplotlib.pyplot as plt
from astropy.io import fits


iteration = 2
segment =20

# Load the FITS file containing the PSF images and OPDs
fits_path = f'psfimgs{iteration}/scanseg{segment}.fits'

with fits.open(fits_path) as hdul:
    psf_images = hdul[0].data  # Image stack
    opds = hdul[1].data       # OPDs


# Load the dark
dark_path = f'psfimgs{iteration}/dark.fits'
with fits.open(dark_path) as dark_hdul:
    dark = dark_hdul[0].data


# Subtract the dark from each PSF image
psf_images_dark_subtracted = psf_images# - dark
# psf_images_dark_subtracted = np.clip(psf_images_dark_subtracted, 0, None)  # Ensure no negative values
# psf_images_dark_subtracted = psf_images_dark_subtracted / np.max(psf_images_dark_subtracted)  # Normalize
# psf_images_dark_subtracted = (psf_images_dark_subtracted * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
# log scale the images 
psf_images_dark_subtracted = np.log1p(psf_images_dark_subtracted)

# crop the images to 64x64
crop_size = 100
center = psf_images_dark_subtracted.shape[1] // 2
half_crop = crop_size // 2
offset = 50
psf_images_dark_subtracted = psf_images_dark_subtracted[:, center-half_crop:center+half_crop+offset, center-half_crop-offset:center+half_crop]

# Create a gif from the PSF images
import imageio
gif_path = f'psfimgs{iteration}/psf_seg{segment}_scan.gif'
# save in viridis colormap
psf_images_dark_subtracted = [plt.cm.magma(img / np.max(img)) for img in psf_images_dark_subtracted]
# convert to uint8
psf_images_dark_subtracted = [(img[:, :, :3] * 255).astype(np.uint8) for img in psf_images_dark_subtracted]
imageio.mimsave(gif_path, psf_images_dark_subtracted, fps=5)   
print(f"GIF saved to {gif_path}")