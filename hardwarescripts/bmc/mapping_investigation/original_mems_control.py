'''
Uses SCExAO SHM
Tries to reproduce the first time I attempted this actuator mapping exercsie. In reality, it shouldn't be much different to shm_mems_control.py. That's just a more refined version.
ONLY difference to original is we save as a npz file not txt. But think about it, the picture itself won't change. It's just the file format that's different.
'''

import ImageStreamIOWrap
from pyMilk.interfacing.shm import SHM  
import numpy as np  
import time
import os


class MEMS():
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
        
    def set_volt_actuator(self, actuator:int, voltage:float) -> None:
        data = self.dmvolt.get_data()
        data[actuator] = voltage
        self.dmvolt.set_data(data)

    def set_volt(self, segment:int, actuator:int, voltage:float) -> None:
        data = self.dmvolt.get_data()
        reshaped_data = data.reshape(37, 3)
        reshaped_data[segment, actuator] = voltage
        updated_data = reshaped_data.reshape(111)

        self.dmvolt.set_data(updated_data)

    def set_ptt_all(self, piston, tip, tilt) -> None:
        # make an ssertion statement of the datatype, range and shape of pisotn, tip, tilt

        data = self.dmptt.get_data()

        reshaped_data = data.reshape(37, 3)
        reshaped_data[:, 0] = piston
        reshaped_data[:, 1] = tip
        reshaped_data[:, 2] = tilt

        updated_data = reshaped_data.reshape(111)
        self.dmptt.set_data(updated_data)
    
    def set_volt_all(self, voltage) -> None:
        data = self.dmvolt.get_data()
        data[:] = voltage

        self.dmvolt.set_data(data)


    def get_volt(self) -> np.ndarray:

        return self.dmvolt.get_data()

    def get_ptt(self) -> np.ndarray:

        return self.dmptt.get_data()
    

    def flatten_ptt(self) -> None:
        flat = np.zeros(111)
        self.dmptt.set_data(flat)
        
    def flatten_volt(self) -> None:

        flat = np.loadtxt('32AW038_1500nm.txt')
        
        self.dmvolt.set_data(flat)
    
    def randomise_ptt(self):
        piston = np.random.rand(37) * 1000
        tip = np.random.rand(37) * 0.001
        tilt = np.random.rand(37) * 0.001

        combined_array = np.column_stack((piston, tip, tilt)).reshape(-1)

        self.dmptt.set_data(combined_array)

    def randomise_volt(self):
        volt = 0.5*np.random.rand(111)
        self.dmvolt.set_data(volt)
    
    
    

if __name__ == "__main__":
    dm = MEMS()

    pupil = SHM('glintpg1')
    image = SHM('glintpg2')
    currentdir = os.getcwd()
    path = currentdir+'/data/frames6/raw/'

    dm.set_volt_all(0)

    for a in np.arange(111):
        dm.set_volt_actuator(a, 0.3)
        time.sleep(0.1)
        p = pupil.get_data()
        i = image.get_data()
        np.savez(path+f'pupil_actuator{a}_volt0.3.npz', p)
        np.savez(path+f'image_actuator{a}_volt0.3.npz', i)
        dm.set_volt_all(0)