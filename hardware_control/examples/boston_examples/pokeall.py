#!/usr/bin/env python

# Before running this, copy the "bmc" module from Python3\site-packages\bmc
#  to your Python installation's site-packages directory.
import bmc
import time

try:
    import numpy as np
    has_numpy = True
except ImportError:
    has_numpy = False

def poke_all_numpy(mirror):
    """
    Poke all actuators using numpy arrays.

    The numpy array must be converted to a list before sending to the DM.
    """
    import numpy as np
        
    data = np.zeros(dm.num_actuators(), dtype=float)
    mirror.send_data(data)
    
    for x in range(mirror.num_actuators()):
        # print('Poking actuator ', x)
        data[x] = 0.5
        mirror.send_data(data.tolist())
        data[x] = 0.0
        time.sleep(0.01)    
    print("Sent ", x+1, " shapes.")
    dm.send_data(data)

def poke_all_cppvector(mirror):
    """Poke all actuators using the swig interface to C++ vectors."""
    data = bmc.DoubleVector()

    data.assign(mirror.num_actuators(), 0.0)
    mirror.send_data(data)
    
    for x in range(mirror.num_actuators()):
        print(x)
        data[x] = 0.5
        mirror.send_data(data)    
        data[x] = 0.0
        time.sleep(0.01)  
    print("Sent ", x+1, " shapes.")
    mirror.send_data(data)


dm = bmc.BmcDm()

# SRB: changed serial number from 'MultiUSB000' to match the one on the DM
serial = '32AW038#027'
err_code = dm.open_dm(serial)
if err_code:
    raise Exception(dm.error_string(err_code))

if (has_numpy):
    print('Using numpy.')
    poke_all_numpy(dm)
else:
    poke_all_cppvector(dm)

dm.close_dm()