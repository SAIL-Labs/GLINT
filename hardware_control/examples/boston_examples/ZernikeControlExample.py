#!/usr/bin/env python
# This Python file uses the following encoding: utf-8

# Before running this, copy the "bmc" module from Python3\site-packages\bmc
#  to your Python installation's site-packages directory.
import bmc
import numpy as np
import time
import os

dm = bmc.BmcDm()
dm.configure_log(os.path.abspath('dmsdk-ol-example.log'), bmc.BMC_LOG_DEBUG);
# dm.set_profiles_path(os.getcwd())
err_code = dm.open_dm('MultiUSBOL1')
if err_code:
    print('Zernike test failed.')
    raise Exception(dm.error_string(err_code))
#err_code = dm.load_calibration_file('Sample_Multi_OLC1_CAL.mat')
#if err_code:
#    print('Zernike test failed.')
#    raise Exception(dm.error_string(err_code))

## Get piston range

# Get full piston range with no tilt
err_code, minPiston, maxPiston = dm.get_segment_range(0, bmc.DM_Piston, 0, 0, 0, True);
if err_code:
    raise Exception(dm.error_string(err_code))

print('DM Piston range: %f to %f nm' % (minPiston, maxPiston))

Zernike_coefficients = [ 0, 0, 0, 0, 0, 0 ]
w = dm.num_actuators_width()    # 12 for Multi, 13 for Multi-C

# Add 200nm RMS of defocus
Zernike_coefficients[4] = 200
surface = bmc.DoubleVector(w*w)
err_code, surface = dm.zernike_surface(Zernike_coefficients, 0, 0)
#print(np.asarray(surface).reshape(w,w))
if err_code:
    raise Exception(dm.error_string(err_code))
err_code = dm.set_surface(surface, w, w)
if err_code:
    raise Exception(dm.error_string(err_code))
time.sleep(0.1)

# Add 100nm RMS of astigmatism
Zernike_coefficients[3] = 100
err_code, surface = dm.zernike_surface(Zernike_coefficients, 0, 0)
if err_code:
    raise Exception(dm.error_string(err_code))
err_code = dm.set_surface(surface, w, w)
if err_code:
    raise Exception(dm.error_string(err_code))
time.sleep(0.1)

# Add 50nm RMS of tilt
Zernike_coefficients[2] = 50
err_code, surface = dm.zernike_surface(Zernike_coefficients, 0, 0)
if err_code:
    raise Exception(dm.error_string(err_code))
err_code = dm.set_surface(surface, w, w)
if err_code:
    raise Exception(dm.error_string(err_code))
time.sleep(0.1)

# Add 1000nm RMS of astigmatism - out of calibrated range
Zernike_coefficients[5] = 1000
err_code, surface = dm.zernike_surface(Zernike_coefficients, 0, 0)
if err_code:
    raise Exception(dm.error_string(err_code))
err_code = dm.set_surface(surface, w, w)
if err_code == bmc.ERR_OUT_OF_LUT_RANGE:
    print('Zernike shape out of range, as expected, setting clipped shape.')
    err_code = dm.set_surface(surface, w, w, options=bmc.DM_SURFACE_BEST_EFFORT)
    if err_code:
        raise Exception(dm.error_string(err_code))
time.sleep(0.1)

print('Zernike test complete.')

dm.close_dm()