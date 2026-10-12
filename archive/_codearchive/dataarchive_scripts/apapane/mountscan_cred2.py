import sys

from chipMountControl import Mount
from pyMilk.interfacing.shm import SHM
import numpy as np
from astropy.io import fits
import glint_paths

apapane = SHM('apapane')
mount = Mount('/dev/serial/by-id/usb-SURUGA_SEIKI_SURUGA_SEIKI_DS102-if00-port0', 38400)

savefile_path = str(glint_paths.data_dir('_dataarchive', 'apapane', 'xyscans'))
dark_filepath = str(glint_paths.DATA_ROOT / '_dataarchive' / 'apapane' / 'darks' / 'dark.fits')
with fits.open(dark_filepath) as hdul:
        dark = hdul[0].data[0]
        dark = np.array(dark, dtype=float)

dict_axes = {'pitch':1, 'roll':2, 'yaw':3, 'x':4, 'z':5, 'y':6}

# Variables
select_axes = ['x','y']
scan_range = np.array([600, 600])
step_size = np.array([8, 8])
start_pos = np.array([2800, 3000])

iteration = 114

h_start = 50
h_end = 270

v_start_1 = 240
v_end_1 = 246
v_start_2 = 221
v_end_2 = 227
v_start_3 = 201
v_end_3 = 207


# Set up
axes = [dict_axes[ax] for ax in select_axes]
res = scan_range/step_size
res = np.ceil(res).astype(int)
photometry_1 = np.zeros(res)
photometry_2 = np.zeros(res)
photometry_3 = np.zeros(res)

# Move mount to start position
for i, ax in enumerate(axes):
    print(mount.get_pos(ax))
    mount.set_pos(ax, start_pos[i])

    while mount.in_motion(ax) or mount.get_pos(ax) != start_pos[i]:
        pass
    
    print(mount.get_pos(ax))

# Scan
for i in range(res[0]):
    print(('{}: {}'.format(select_axes[0], mount.get_pos(axes[0])), '{}: {}'.format(select_axes[1], mount.get_pos(axes[1]))))

    for j in range(res[1]):

        # Subtract dark
        data = apapane.get_data() - dark

        photometry_1[i,j] = np.sum(data[v_start_1:v_end_1, h_start:h_end])
        photometry_2[i,j] = np.sum(data[v_start_2:v_end_2, h_start:h_end])
        photometry_3[i,j] = np.sum(data[v_start_3:v_end_3, h_start:h_end])

        
        
        # Update axis 2
        prev_pos2 = mount.get_pos(axes[1])
        new_pos2 = prev_pos2 + step_size[1]
        mount.set_pos(axes[1], new_pos2)

        while mount.in_motion(axes[1]) or mount.get_pos(axes[1]) != new_pos2:
            pass

    # Update axis 1
    prev_pos1 = mount.get_pos(axes[0])
    new_pos1 = prev_pos1 + step_size[0]
    mount.set_pos(axes[0], new_pos1)

    # Return axis 2 to start position
    mount.set_pos(axes[1], start_pos[1])

    while mount.in_motion(axes[0]) or mount.get_pos(axes[0]) != new_pos1 or mount.in_motion(axes[1]) or mount.get_pos(axes[1]) != start_pos[1]:
        pass

    

# Save data
hdu = fits.PrimaryHDU(photometry_1)
hdul = fits.HDUList([hdu])
hdul.writeto('{}{}scan_spectra1_{}.fits'.format(select_axes[0], select_axes[1],iteration), overwrite=True)

hdu = fits.PrimaryHDU(photometry_2)
hdul = fits.HDUList([hdu])
hdul.writeto('{}{}scan_spectra2_{}.fits'.format(select_axes[0], select_axes[1],iteration), overwrite=True)

hdu = fits.PrimaryHDU(photometry_3)
hdul = fits.HDUList([hdu])
hdul.writeto('{}{}scan_spectra3_{}.fits'.format(select_axes[0], select_axes[1],iteration), overwrite=True)


   