import sys

#!/usr/bin/env python

# Before running this, copy the "bmc" module from Python3\site-packages\bmc
#  to your Python installation's site-packages directory.
import bmc
import time
import os

dm = bmc.BmcDm()
dm.configure_log(os.path.abspath('dmsdk-example.log'), bmc.BMC_LOG_DEBUG);
# dm.set_profiles_path(os.getcwd())
serial = '32AW038#027' # changed from 'HexW111#USB'
if dm.open_dm(serial):
    print('Segment test fail.')
    exit(1)

LUTfilename = 'LUT_32AW038#027.mat'  # changed from 'Sample_Lookup_Table.mat'
dm.load_calibration_file(LUTfilename)

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
    time.sleep(0.01)
    print('Tilt segment %d of %d %fr.' % (k, numSegments, tiltValue))
    dm.set_segment(k, pistonValue, tiltValue, 0, True, True);
    dm.set_segment(k, pistonValue, tiltValue, tiltValue, True, True);
    dm.set_segment(k, pistonValue, 0, tiltValue, True, True);
    time.sleep(0.01)

## Try to piston out of calibrated range
pistonValue = minPiston - 1
segment_no = 5
err_code = dm.set_segment(segment_no, pistonValue, 0, 0, True, True)
if err_code == bmc.ERR_OUT_OF_LUT_RANGE:
    err_code, minPiston, maxPiston = dm.get_segment_range(0, bmc.DM_Piston, 0, 0, 0, True)
    print('Piston %d out of range: [%d, %d]\n' % (pistonValue, minPiston, maxPiston))
else:
    raise Exception(dm.error_string(err_code))

## Piston or tilt all segments simultaneously
pistonValue = (maxPiston + minPiston)/2;
tiltValue = -1.0e-6;
numSegments = int(dm.num_actuators()/3);
sendNow = False;

for k in range(numSegments):
    print('Piston/tilt segment %d of %d %fnm %fr.' % (k, numSegments, pistonValue, tiltValue))
    if (k == numSegments):
        sendNow = True;
    dm.set_segment(k, pistonValue, tiltValue, tiltValue, True, sendNow);

print('Segment test complete.')

dm.close_dm()