#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Before running this, copy the "bmc" module from Python3\site-packages\bmc
#  to your Python installation's site-packages directory.
import bmc
import numpy as np
import time

def start_sequence(dm):
    #Configure/Enable Sequencing

    delay = .1
    frame_len = dm.num_actuators()
    seq_len = 4096
    num_frames = seq_len // frame_len
    seq = np.zeros(frame_len*num_frames)
    for frame in range(num_frames):
        seq[(frame) * frame_len + frame] = 0.6
        seq[(frame) * frame_len + (frame + 1)] = 0.8

    err_code = dm.configure_sequence(seq, delay, frame_len, num_frames)
    if err_code:
        raise Exception(dm.error_string(err_code))
    print("configured sequence")
    err_code = dm.enable_sequence(100, True)
    if err_code:
        raise Exception(dm.error_string(err_code))
    print("enabled sequence")

def start_dither(dm):
    #Configure/Enable Dithering
    x = np.linspace(0, np.pi)
    waveform = np.sin(x) #values [0,1]
    gains = np.ones(dm.num_actuators())  #actuators to be dithered

    err_code = dm.configure_dither(waveform, gains)
    if err_code:
        raise Exception(dm.error_string(err_code))
    print("configured dithering")

    err_code = dm.enable_dither(100, True)
    if err_code:
        raise Exception(dm.error_string(err_code))
    print("enabled dithering")

if __name__ == '__main__':

    dm = bmc.BmcDm()
    err_code = dm.open_dm('HVA140_0000')
    if err_code:
        raise Exception(dm.error_string(err_code))


    start_sequence(dm)
    time.sleep(15)
    dm.enable_sequence(0, False)
    print('disabled sequencing')

    start_dither(dm)
    time.sleep(15)
    dm.enable_dither(0, False)
    print('disabled dithering')
    dm.close_dm()
