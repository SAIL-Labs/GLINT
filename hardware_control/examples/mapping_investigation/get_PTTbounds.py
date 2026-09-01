import sys

import bmc
from datetime import datetime 
import os


dm = bmc.BmcDm()

# dm.set_profiles_path(os.getcwd())
serial = '32AW038#027' # changed from 'HexW111#USB'
if dm.open_dm(serial):
    print('Segment test fail.')
    exit(1)



## Get piston and tilt ranges
try:
    seg = 31
    print("Segment:", seg)

    # Get full piston range with no tilt
    err_code, minPiston, maxPiston = dm.get_segment_range(seg, bmc.DM_Piston, 0, 0, 0, True)
    if err_code:
        print(err_code)
        raise Exception(dm.error_string(err_code))


    # Get X-Tilt range for minimum piston
    err_code, minXTilt, maxXTilt = dm.get_segment_range(seg, bmc.DM_XTilt, minPiston, 0, 0, True)
    print(minXTilt)
    print(maxXTilt)
    if err_code:
        print(err_code)
        raise Exception(dm.error_string(err_code))

    # # Get Y-Tilt range for minimum piston
    err_code, minYTilt, maxYTilt = dm.get_segment_range(seg, bmc.DM_YTilt, minPiston, 0, 0, True)
    if err_code:
        print(err_code)
        raise Exception(dm.error_string(err_code))


    path = os.getcwd()
    filename = 'PTT_bounds.txt'
    file_path = f"{path}/{filename}"
    f = open(file_path, "a")

    dt = str(datetime.now())
    line = f"{dt}\n Segment: {seg}, Min Pist: {minPiston}, Max Pist: {maxPiston}\nMin X Tilt: {minXTilt}, Max X Tilt: {maxXTilt}\nMin Y Tilt: {minYTilt}, Max Y Tilt: {maxYTilt}\n"
    f.writelines(line)
except Exception as e:
    print(e)
    dm.close_dm()
    print("dm closed successfully")