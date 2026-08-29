'''
This file differs to the other exmaples in this directory in that it uses the shared memory of SCExAO. 
It differs to memsControl in the control-code folder in that this file only includes methods that are useful, and has commented code
which can be used in a scexao5 ipython terminal to save actuator mappings.
'''

import ImageStreamIOWrap
from pyMilk.interfacing.shm import SHM  
import numpy as np  
import time
import os

class MEMS():
    '''
    Class to control the MEMS DM.
    '''
    def __init__(self) -> None:
        self.dmvolt = SHM('dmvolt')  
        self.dmptt = SHM('dmptt') 
    
    def set_ptt(self, segment:int, piston:float, tip:float, tilt:float) -> None:
        data = self.dmptt.get_data()
        reshaped_data = data.reshape(37, 3)

        reshaped_data[segment, 0] = piston
        reshaped_data[segment, 1] = tip
        reshaped_data[segment, 2] = tilt

        updated_data = reshaped_data.reshape(111)
        self.dmptt.set_data(updated_data)
        
    def set_volt(self, actuator:int, voltage:float) -> None:
        data = self.dmvolt.get_data()
        data[actuator] = voltage
        self.dmvolt.set_data(data)

    def get_volt(self) -> np.ndarray:
        return self.dmvolt.get_data()

    def get_ptt(self) -> np.ndarray:
        return self.dmptt.get_data()

    def flatten_ptt(self) -> None:
        n = self.num_actuators()
        flat = np.zeros(n)
        self.dmptt.set_data(flat)
        
    def flatten_volt(self) -> None:
        n = self.num_actuators()
        flat = np.zeros(n)
        self.dmvolt.set_data(flat)

    def num_actuators(self) -> int:
        return 111  # Should be the same as len(self.get_volt)

    def num_segments(self) -> int:
        return 37
    


if __name__ == "__main__":

    dm = MEMS()

    pupil = SHM('glintpg1')
    image = SHM('glintpg2')
    currentdir = os.getcwd()


    path = currentdir+'/data/frames7/raw/'

    num_actuators = dm.num_actuators()
    dm.flatten_volt()
    
    for x in range(num_actuators):
        dm.set_volt(x, 0.3)
        
        p = pupil.get_data()
        i = image.get_data()
        
        np.savez(path+f'pupil_actuator{x}_volt0.3', p)
        np.savez(path+f'image_actuator{x}_volt0.3', i)

        dm.flatten_volt()
        time.sleep(0.01)  
    




    path = currentdir+'/data/frames9/raw/'

    num_segments = dm.num_segments()
    dm.flatten_ptt()

    # need to find a robust way to retrieve the max Piston and min Piston but for now:
    
    minPiston = -2904.2768742257467
    maxPiston = 95.72312577425345
    minXtilt = -0.0038081343797035016
    maxXtilt = 0.0032421492281229695
    minYtilt = -0.004974432194510029
    maxYtilt = 0.0029015603043231466

    pistonValue = (maxPiston + minPiston)/2

    tiltValue = -1.0e-6 #  idk why, it was given in the Boston example code
    numSegments = dm.num_segments()

    for k in range(num_segments):
        dm.set_ptt(k, pistonValue, 0, 0)
        p = pupil.get_data()
        i = image.get_data()
        np.savez(path+f'pupil_pist_segment{k}', p)
        np.savez(path+f'image_pist_segment{k}', i)
        time.sleep(0.01)

        dm.set_ptt(k, pistonValue, tiltValue, 0)
        p = pupil.get_data()
        i = image.get_data()
        np.savez(path+f'pupil_tip_segment{k}', p)
        np.savez(path+f'image_tip_segment{k}', i)
        time.sleep(0.01)

        dm.set_ptt(k, pistonValue, tiltValue, tiltValue)
        p = pupil.get_data()
        i = image.get_data()
        np.savez(path+f'pupil_tilt_segment{k}', p)
        np.savez(path+f'image_tilt_segment{k}', i)
        time.sleep(0.01)

        dm.flatten_ptt()

