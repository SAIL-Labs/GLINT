import sys
sys.path.append('/home/scexao/steph/control-code/')

from chipMountControl import Mount
from pyMilk.interfacing.shm import SHM
import numpy as np
# import matplotlib.pyplot as plt
from astropy.io import fits


apapane = SHM('apapane')
mount = Mount('/dev/serial/by-id/usb-SURUGA_SEIKI_SURUGA_SEIKI_DS102-if00-port0', 38400)

savefile_path = '/home/scexao/steph/spectrograph/scans/'
dark_path = '/home/scexao/steph/spectrograph/darks/'
dark_filename = 'apapane_2024-07-15_18:29:32.213198.fits'
dark = fits.open(dark_path + dark_filename)[0].data
dark = dark.astype(np.float64) # Convert to float64 to ensure subtraction works i.e. doesn't wrap around

dict_axes = {'pitch':1, 'roll':2, 'yaw':3, 'x':4, 'z':5, 'y':6}

# Variables
select_axes = ['x','y']
scan_range = np.array([120, 120])
step_size = np.array([4, 4])
start_pos = np.array([3080, 3230])

# Z scan
zstart_pos = 1900
zscan_range = 350
zstep_size = 50

iteration = 7


v_start = 160
v_end = -1

h_start_1 = 83
h_end_1 = 97
h_start_2 = 113
h_end_2 = 127
h_start_3 = 145
h_end_3 = 159


# Set up
axes = [dict_axes[ax] for ax in select_axes]
res = scan_range/step_size
res = np.ceil(res).astype(int)

zres = zscan_range/zstep_size
zres = np.ceil(zres).astype(int)

frame = 23
z_axis = 5

# Move mount to start position
mount.set_pos(z_axis, zstart_pos)

while mount.in_motion(z_axis) or mount.get_pos(z_axis) != zstart_pos:
    pass

# Set previous z position
prev_z = mount.get_pos(z_axis)

# Scan
for z in range(zres):
    print('z: {}'.format(mount.get_pos(5)))

    photometry_1 = np.zeros(res)
    photometry_2 = np.zeros(res)
    photometry_3 = np.zeros(res)

    # Move mount to start position for x and y
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

            # Get data first before moving
            photometry_1[i,j] = np.sum(apapane.get_data()[v_start:v_end, h_start_1:h_end_1])
            photometry_2[i,j] = np.sum(apapane.get_data()[v_start:v_end, h_start_2:h_end_2])
            photometry_3[i,j] = np.sum(apapane.get_data()[v_start:v_end, h_start_3:h_end_3])
            
            
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
    hdul.writeto('{}{}scan_spectra1_frame{}_{}.fits'.format(select_axes[0], select_axes[1],frame,iteration), overwrite=False)

    hdu = fits.PrimaryHDU(photometry_2)
    hdul = fits.HDUList([hdu])
    hdul.writeto('{}{}scan_spectra2_farme{}_{}.fits'.format(select_axes[0], select_axes[1],frame,iteration), overwrite=False)

    hdu = fits.PrimaryHDU(photometry_3)
    hdul = fits.HDUList([hdu])
    hdul.writeto('{}{}scan_spectra3_frame{}_{}.fits'.format(select_axes[0], select_axes[1],frame,iteration), overwrite=False)

    # Update z axis
    prev_z = mount.get_pos(z_axis)
    new_z = prev_z + zstep_size
    mount.set_pos(z_axis, new_z)

    while mount.in_motion(z_axis) or mount.get_pos(z_axis) != new_z:
        pass
    
    frame += 1