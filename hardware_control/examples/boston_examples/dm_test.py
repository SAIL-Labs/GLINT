#!/usr/bin/env python

# Before running this, copy the "bmc" module from Python3\site-packages\bmc
#  to your Python installation's site-packages directory.
import sys
import bmc

dm = bmc.BmcDm()

# SRB: changed serial number from 'MultiUSB000' to match the one on the DM
serial = '32AW038#027'

err_code = dm.open_dm(serial)
if err_code:
    raise Exception(dm.error_string(err_code))

mapping = list(dm.default_mapping())

data = bmc.DoubleVector()
data.assign(dm.num_actuators(), 0.0)
dm.send_data(data)

monotonic_map = range(0, dm.num_actuators())
dm.send_data_custom_mapping(data, monotonic_map)

print('BMC error status:', dm.error_string(dm.get_status()))

dm.close_dm()
