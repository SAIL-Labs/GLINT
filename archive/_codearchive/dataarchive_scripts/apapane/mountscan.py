import sys

from chipMountControl import Mount
from pyMilk.interfacing.shm import SHM
import numpy as np
# import matplotlib.pyplot as plt
from astropy.io import fits
import glint_paths

apapane = SHM('apapane')
mount = Mount('/dev/serial/by-id/usb-SURUGA_SEIKI_SURUGA_SEIKI_DS102-if00-port0', 38400)

# savefile_path = f'{glint_paths.DATA_ROOT}/_dataarchive/spectrograph/scans/'
dark_path = str(glint_paths.DATA_ROOT / '_dataarchive' / 'apapane' / 'darks') + '/'
dark_filename = 'apapane_2024-07-24_23:28:16.727510.fits'
dark = fits.open(dark_path + dark_filename)[0].data
dark = dark.astype(np.float64) # Convert to float64 to ensure subtraction works i.e. doesn't wrap around

dict_axes = {'pitch':1, 'roll':2, 'yaw':3, 'x':4, 'z':5, 'y':6}

# Variables
select_axes = ['x','y']
# (340, 340)
# scan_range = np.array([400, 400])
# scan_range = np.array([340, 340])
# (4, 4)
# step_size = np.array([20, 20])
# step_size = np.array([4, 4])
# (2980, 3225) 
# start_pos = np.array([2930, 3220])
# start_pos = np.array([2980, 3225])
scan_range = np.array([450, 450])
step_size = np.array([10, 10])
start_pos = np.array([2900, 3150])

iteration = 111

v_start = 120
v_end = -1

h_start_1 = 87
h_end_1 = 97
h_start_2 = 118
h_end_2 = 128
h_start_3 = 149
h_end_3 = 159


# Set up
axes = [dict_axes[ax] for ax in select_axes]
res = scan_range/step_size
res = np.ceil(res).astype(int)
# photometry_1 = np.zeros(res)
# photometry_2 = np.zeros(res)
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
        data = apapane.get_data() 

        # photometry_1[i,j] = np.sum(data[v_start:v_end, h_start_1:h_end_1])
        # photometry_2[i,j] = np.sum(data[v_start:v_end, h_start_2:h_end_2])
        photometry_3[i,j] = np.sum(data[v_start:v_end, h_start_3:h_end_3])

        # Get data first before moving
        # photometry_1[i,j] = np.sum(apapane.get_data()[v_start:v_end, h_start_1:h_end_1])
        # photometry_2[i,j] = np.sum(apapane.get_data()[v_start:v_end, h_start_2:h_end_2])
        # photometry_3[i,j] = np.sum(apapane.get_data()[v_start:v_end, h_start_3:h_end_3])
        
        
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
# hdu = fits.PrimaryHDU(photometry_1)
# hdul = fits.HDUList([hdu])
# hdul.writeto('{}{}scan_spectra1_{}.fits'.format(select_axes[0], select_axes[1],iteration), overwrite=True)

# hdu = fits.PrimaryHDU(photometry_2)
# hdul = fits.HDUList([hdu])
# hdul.writeto('{}{}scan_spectra2_{}.fits'.format(select_axes[0], select_axes[1],iteration), overwrite=True)

hdu = fits.PrimaryHDU(photometry_3)
hdul = fits.HDUList([hdu])
hdul.writeto('{}{}scan_spectra3_{}.fits'.format(select_axes[0], select_axes[1],iteration), overwrite=True)


   