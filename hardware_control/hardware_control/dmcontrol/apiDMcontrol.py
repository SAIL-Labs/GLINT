from pathlib import Path

from . import bmc
import numpy as np

FLATS_DIR = Path(__file__).resolve().parent / 'flats'

DM_Piston = bmc.DM_Piston
DM_XTilt = bmc.DM_XTilt
DM_YTilt = bmc.DM_YTilt

class DM():
    def __init__(self, serial) -> None:
        self.dm = bmc.BmcDm()
        self.serial = serial
    
    def openDM(self) -> None:
        err_code = self.dm.open_dm(self.serial)
        
        if err_code:
            raise Exception(self.dm.error_string(err_code))
    
    def closeDM(self) -> None:
        self.dm.close_dm()

    def num_actuators(self) -> int:
        return self.dm.num_actuators()

    def send_data(self, data:np.ndarray) -> None:
        '''
        Sends a full array of actuator vaues to the DM.
        '''
        err_code = self.dm.send_data(data.tolist())
        if err_code:
            raise Exception(self.dm.error_string(err_code))
    
    def set_actuator(self, actuator: 'int', data: 'double') -> "int":
        '''
        Set the value of a single actuator.
        '''
        return self.dm.poke(actuator, data)
    
    def set_segment(self, segment: 'int', piston: 'double', xTilt: 'double', yTilt: 'double') -> "int":
        '''
        Set the PTT values of a segment. Piston is in nm, tilt is in mrad.
        '''
        # Convert to radians
        xTilt = xTilt / 1000
        yTilt = yTilt / 1000
        err_code =  self.dm.set_segment(segment, piston, xTilt, yTilt, True, True)

        if err_code == bmc.ERR_OUT_OF_LUT_RANGE:
            err_string = f"Out of range! [{piston}, {xTilt}, {yTilt}]\n"
            print(err_string)
            return -1
        elif err_code == 0:
            # print('Segment set successfully')
            pass 
        else:
            raise Exception(self.dm.error_string(err_code))

    
    def get_actuator_data(self) -> np.ndarray:
        return np.array(self.dm.get_actuator_data())
    
    def get_segment_range(self, segment: 'int', axis: str, piston: 'double', xTilt: 'double', yTilt: 'double', applyOffsets = True) -> "int":
        if axis == 'piston':
            axis = bmc.DM_Piston
        elif axis == 'tip':
            axis = bmc.DM_XTilt
        elif axis == 'tilt':
            axis = bmc.DM_YTilt

        
        return self.dm.get_segment_range(segment, axis, piston, xTilt, yTilt, applyOffsets)
    
    def flatten(self) -> None:
        flat = np.loadtxt(FLATS_DIR / '32AW038_1500nm.txt', dtype=float)
        print('Flattening DM')
        self.send_data(flat)

    def select_flatten(self, wavelength = 1500) -> None:
        flat = np.loadtxt(FLATS_DIR / f'32AW038_{wavelength}nm.txt', dtype=float)
        self.send_data(flat)

    def sendzeros(self) -> None:
        zeros = np.zeros(111)
        print('Sending zeros')
        self.send_data(zeros)
    
    
    
    


# if __name__ == "__main__":
#     dm = DM('32AW038#027')
#     dm.openDM()
#     print("Number of actuators:", dm.num_actuators())
#     dm.closeDM()
#     print('DM opened and closed successfully')

    
