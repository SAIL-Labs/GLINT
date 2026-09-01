import json
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import imageio
import os
import glint_paths

# Load settings
with open('scanparameters.json', 'r') as f:
    params = json.load(f)

# Access parameters
date = params['date']
iteration = 2
select_axes = params['select_axes']
xystep_size = np.array(params['xystep_size'])
xyscan_range = np.array(params['xyscan_range'])
xystart_pos = np.array(params['xystart_pos'])
zstep_size = np.array(params['zstep_size'])
zscan_range = 350#np.array(params['zscan_range'])
zstart_pos = 1750#np.array(params['zstart_pos'])

path = f'{glint_paths.data_dir("alignment_scans", "zscans", "2025", date, f"scan{iteration}")}/'
os.makedirs(f'{path}imgs', exist_ok=True)

zsteps = np.ceil(zscan_range / zstep_size).astype(int) + 1
zpositions = np.linspace(zstart_pos, zstart_pos + zscan_range, zsteps)

stepsize = xystep_size
scanrange = xyscan_range
startx, starty = xystart_pos
endx = startx + (scanrange[0] - stepsize[0])
endy = starty + (scanrange[1] - stepsize[1])

# First find global maximum pixel across all spectra and all frames
maxpixel = 0
for spectra in [1, 2, 3]:
    for frame in range(zsteps):
        filename = f'{path}zscan_spectra{spectra}_frame{frame}_{iteration}.fits'
        data = fits.getdata(filename)
        maxpixel = max(maxpixel, np.max(data))

# Now generate subplots for each frame
for frame in range(zsteps):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    z_pos = int(zpositions[frame])
    
    for i, spectra in enumerate([1, 2, 3]):
        filename = f'{path}zscan_spectra{spectra}_frame{frame}_{iteration}.fits'
        data = fits.getdata(filename) / maxpixel

        ax = axes[i]
        im = ax.imshow(data, vmin=0, vmax=1, cmap='viridis', origin='lower',
                       extent=[starty - stepsize[1]/2, endy - stepsize[1]/2,
                               startx - stepsize[0]/2, endx - stepsize[0]/2])
        ax.set_title(f"Spectra {spectra}")
    
    fig.suptitle(f"Z pos: {z_pos}")
    # fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{path}imgs/zscan_all_spectra_frame{frame}_{iteration}.png')
    plt.close()

# Create GIF
image_paths = [f'{path}imgs/zscan_all_spectra_frame{frame}_{iteration}.png' for frame in range(zsteps)]
images = [imageio.imread(image_path) for image_path in image_paths]
gif_path = f'{path}imgs/zscan_all_spectra_combined.gif'
imageio.mimsave(gif_path, images, fps=3)
