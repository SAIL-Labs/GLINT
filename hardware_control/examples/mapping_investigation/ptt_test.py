import sys
sys.path.append('/home/scexao/steph/bmc')

import bmc
import time
import os
import numpy as np
from pyMilk.interfacing.shm import SHM


pupil = SHM('glintpg1')
image = SHM('glintpg2')
path = '/home/scexao/steph/bmc/mapping_investigation/data/frames5'


dm = bmc.BmcDm()

# dm.set_profiles_path(os.getcwd())
serial = '32AW038#027' # changed from 'HexW111#USB'
if dm.open_dm(serial):
    print('Segment test fail.')
    exit(1)

## Get piston and tilt ranges

# Get full piston range with no tilt
err_code, minPiston, maxPiston = dm.get_segment_range(0, bmc.DM_Piston, 0, 0, 0, True);
if err_code:
    raise Exception(dm.error_string(err_code))

# Get X-Tilt range for minimum piston
err_code, minXTilt, maxXTilt = dm.get_segment_range(0, bmc.DM_XTilt, minPiston, 0, 0, True);
if err_code:
    raise Exception(dm.error_string(err_code))

# Get Y-Tilt range for minimum piston
err_code, minYTilt, maxYTilt = dm.get_segment_range(0, bmc.DM_YTilt, minPiston, 0, 0, True);
if err_code:
    raise Exception(dm.error_string(err_code))

    ## Piston or tilt one segment at a time
pistonValue = (maxPiston + minPiston)/2;
tiltValue = -1.0e-6;
numSegments = int(dm.num_actuators()/3);

for k in range(numSegments):
    print('Piston segment %d of %d %fnm.' % (k, numSegments, pistonValue))
    dm.set_segment(k, pistonValue, 0, 0, True, True);
    p = pupil.get_data()
    i = image.get_data()
    np.savez(path+f'pupil_pist_segment{k}', p)
    np.savez(path+f'image_pist_segment{k}', i)
    time.sleep(0.01)

    dm.set_segment(k, pistonValue, tiltValue, 0, True, True);
    p = pupil.get_data()
    i = image.get_data()
    np.savez(path+f'pupil_tip_segment{k}', p)
    np.savez(path+f'image_tip_segment{k}', i)
    time.sleep(0.01)

    dm.set_segment(k, pistonValue, tiltValue, tiltValue, True, True);
    p = pupil.get_data()
    i = image.get_data()
    np.savez(path+f'pupil_tilt_segment{k}', p)
    np.savez(path+f'image_tilt_segment{k}', i)
    time.sleep(0.01)

print('Segment test complete.')

dm.close_dm()