import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pyMilk.interfacing.shm import SHM


apapane = SHM('apapane')

frame = apapane.get_data()

nullpeaks = [141, 235, 360]
null = nullpeaks[0]
box_halfwidth = 2
top, bottom = 10,50

bright = np.mean(frame[null-box_halfwidth:null+box_halfwidth,top:bottom])

nullshifted = null + 10
dark = np.mean(frame[nullshifted-box_halfwidth:nullshifted+box_halfwidth,top:bottom])

snr = bright/dark

print(snr)



