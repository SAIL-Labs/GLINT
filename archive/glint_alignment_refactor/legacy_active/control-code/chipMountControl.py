from serial import Serial
import os
from datetime import datetime

class Mount:

    def __init__(self, port, baudrate) -> None:
        # self.s = Serial('/dev/ttyUSB2')#, baudrate=38400) 
        self.s = Serial(port, baudrate)

        # path = os.getcwd()
        path = '/home/scexao/glint/control-code'
        filename = 'command_log.txt'
        file_path = f"{path}/{filename}"

        self.f = open(file_path, "a")
    
    def closeFile(self):
        self.f.close()
    
    
    def read_command(self, cmd: str) -> str:
        '''
        This function sends a command to the stage and returns the response which is used to create verbose functions useable in python.
        params: cmd: str - the command to send to the stage
        '''
        self.update_log(cmd)

        with self.s as ser:
            ser.write(f'{cmd}\r'.encode('ascii'))
            ret = ser.read_until(b'\r')[:-1].decode()
            self.update_log(ret)
            return ret
    
    def send_command(self, cmd: str) -> str:
        '''
        This function sends a command to the stage and returns the response which is used to create verbose functions useable in python.
        params: cmd: str - the command to send to the stage
        '''
        with self.s as ser:
            ser.write(f'{cmd}\r'.encode('ascii'))
        
        self.update_log(cmd)

        

    
    def rstop(self) -> None:
        '''
        This function reduction stops all stages.
        '''
        cmd = 'STOP 1'
        self.send_command(cmd)

    def estop(self) -> None:
        '''
        This function emergency stops all stages.
        '''
        cmd = 'STOP 0'
        self.send_command(cmd)

 
    def idn(self) -> str:
        '''
        This function returns the identification string of the stage.
        '''
        cmd = '*IDN?'
        ret = self.read_command(cmd)
        return ret
        
    def get_pos(self, axis:int) -> int:
        '''
        This function returns the current position of the stage.
        '''
        cmd = f'AXI{axis}:POS?'
        pos = int(self.read_command(cmd))
        return pos
    
    def set_pos(self, axis:int, pos: int) -> int:
        '''
        This function moves the stage to the specified position.
        params: pos: int - the position to move to
        '''
        cmd = f'AXI{axis}:GOABS {pos}'
        self.send_command(cmd)



    
    def get_speed(self, axis:int) -> int:
        '''
        This function returns the current speed of the stage.
        returns: int - the speed of the stage.
        '''
        cmd = f'AXI{axis}:SELSP?'
        speed = int(self.read_command(cmd))
        return speed
    
    def set_speed(self, axis:int, speed: int) -> None:
        '''
        This function sets the speed of the stage.
        params: speed: int - the speed to set the stage to. This is an integer between 1 and 9 where 1 is the slowest speed and 9 is the fastest speed.
        '''
        # assert speed > 0 and speed < 10 and isinstance(speed, int)
        cmd = f'AXI{axis}:SELSP {speed}'
        self.send_command(cmd)

        

    def set_home(self, axis:int, home: int) -> None:
        '''
        This function sets the home position of the stage.
        params: home: int - the position to set the home position to.
        '''
        cmd = f'AXI{axis}:HOMEP {home}'
        self.send_command(cmd)

    
    def get_home(self, axis:int) -> int:
        '''
        This function returns the home position of the stage.
        returns: int - the home position
        '''
        cmd = f'AXI{axis}:HOMEP?'
        ret = int(self.read_command(cmd))
        return ret 

    def is_home(self, axis:int) -> bool:
        '''
        This function returns whether the stage is at the home position.
        returns: bool - True if the stage is at the home position, False if the stage is not at the home position.
        '''
        cmd = f'AXI{axis}:HOME?'
        ret = bool(int(self.read_command(cmd)))
        return ret
    
    def go_home(self, axis:int) -> None:
        '''
        This function moves the stage to the home position.
        '''
        cmd = f'AXI{axis}:GO 3'
        self.send_command(cmd)

    
    def set_origin_pattern(self, axis:int, direction:int) -> None:
        '''
        This function moves the stage to the origin.
        params: direction: int - 3 for positive direction, 4 for negative direction.
        '''
        cmd = f'AXI{axis}:MEMSW0 {direction}'
        self.send_command(cmd)

    
    def get_origin_pattern(self, axis:int) -> int:
        '''
        This function returns the direction of the origin.
        returns: int - 3 for positive direction, 4 for negative direction.
        '''
        cmd = f'AXI{axis}:MEMSW0?'
        ret = int(self.read_command(cmd))
        return ret
    
    def go_origin(self, axis:int) -> None:
        '''
        This function moves the stage to the origin.
        '''
        cmd = f'AXI{axis}:GO 2'
        self.send_command(cmd)
    
    def is_origin(self, axis:int) -> bool:
        '''
        This function returns whether the stage is at the origin.
        returns: bool - True if the stage is at the origin, False if the stage is not at the origin.
        '''
        cmd = f'AXI{axis}:ORG?'
        ret = bool(int(self.read_command(cmd)))
        return ret
    
    
    def is_limit(self, axis:int, soft = False) -> str:

        '''

        params: soft: bool - if True, check soft limits, else check hard limits
        returns: str - "Undetected", "Detected upper limit", "Detected lower limit", "Detected upper & lower limits"
        '''
        if soft:
            cmd = f'AXI{axis}:SLIMIT?'
        elif not soft:
            cmd = f'AXI{axis}:LIMIT?'
        else:
            raise ValueError('soft must be True or False')
        
        limit = int(self.read_command(cmd))

        if limit == 0:
            ret = "Undetected"
        elif limit == 1:
            ret = "Detected upper (CW) limit"
        elif limit == 2:
            ret = "Detected lower (CCW) limit"
        elif limit == 3:
            ret = "Detected upper (CW) & lower (CCW) limits"
        else:
            raise ValueError(f'returned unidentified limit state {limit}, must be 0, 1, 2 or 3')
        return ret

    def enable_lim(self, axis:int, direction:str) -> None:
        '''
        This function enables the limit in the specified direction
        params: direction: str - "CW" or "CCW" where "CW" is the positive direction and "CCW" is the negative direction.
        '''

        if direction == 'CW':
            cmd = f'AXI{axis}:CWSLE 1'
        elif direction == 'CCW':
            cmd = f'AXI{axis}:CCWSLE 1'
        else:
            raise ValueError('direction must be "CW" or "CCW"')
        
        self.send_command(cmd)

    
    def disable_lims(self, axis:int, direction:str) -> None:
        '''
        This function disables the limit in the specified direction
        params: direction: str - "CW" or "CCW" where "CW" is the positive direction and "CCW" is the negative direction.
        '''

        if direction == 'CW':
            cmd = f'AXI{axis}:CWSLE 0'
        elif direction == 'CCW':
            cmd = f'AXI{axis}:CCWSLE 0'
        else:
            raise ValueError('direction must be "CW" or "CCW"')
        
        self.send_command(cmd)


    def get_lims(self) -> None:

        '''
        This function returns the limits of the mount.
        returns: str - a string describing the limits of the mount.
        '''

        # Check if each of the 6 axes have either a CW or CCW limit. Then will store the values of those limits in a variable, else it will be None. Finally it will store a string describing these.

        ax1_hasCWlim  = self.lim_enabled(1, 'CW') 
        ax1_hasCCWlim = self.lim_enabled(1, 'CCW')
        ax2_hasCWlim  = self.lim_enabled(2, 'CW')
        ax2_hasCCWlim = self.lim_enabled(2, 'CCW')
        ax3_hasCWlim  = self.lim_enabled(3, 'CW')
        ax3_hasCCWlim = self.lim_enabled(3, 'CCW')
        ax4_hasCWlim  = self.lim_enabled(4, 'CW')
        ax4_hasCCWlim = self.lim_enabled(4, 'CCW')
        ax5_hasCWlim  = self.lim_enabled(5, 'CW')
        ax5_hasCCWlim = self.lim_enabled(5, 'CCW')
        ax6_hasCWlim  = self.lim_enabled(6, 'CW')
        ax6_hasCCWlim = self.lim_enabled(6, 'CCW')

        ax1_CWlim  = self.get_lim(1, 'CW')
        ax1_CCWlim = self.get_lim(1, 'CCW')
        ax2_CWlim  = self.get_lim(2, 'CW')
        ax2_CCWlim = self.get_lim(2, 'CCW')
        ax3_CWlim  = self.get_lim(3, 'CW')
        ax3_CCWlim = self.get_lim(3, 'CCW')
        ax4_CWlim  = self.get_lim(4, 'CW')
        ax4_CCWlim = self.get_lim(4, 'CCW')
        ax5_CWlim  = self.get_lim(5, 'CW')
        ax5_CCWlim = self.get_lim(5, 'CCW')
        ax6_CWlim  = self.get_lim(6, 'CW')
        ax6_CCWlim = self.get_lim(6, 'CCW')

        ret = f"Axis 1: CW limit ({ax1_hasCWlim}): {ax1_CWlim}, CCW limit ({ax1_hasCCWlim}): {ax1_CCWlim} \n"
        ret += f"Axis 2: CW limit ({ax2_hasCWlim}): {ax2_CWlim}, CCW limit ({ax2_hasCCWlim}): {ax2_CCWlim} \n"
        ret += f"Axis 3: CW limit ({ax3_hasCWlim}): {ax3_CWlim}, CCW limit ({ax3_hasCCWlim}): {ax3_CCWlim} \n"
        ret += f"Axis 4: CW limit ({ax4_hasCWlim}): {ax4_CWlim}, CCW limit ({ax4_hasCCWlim}): {ax4_CCWlim} \n"
        ret += f"Axis 5: CW limit ({ax5_hasCWlim}): {ax5_CWlim}, CCW limit ({ax5_hasCCWlim}): {ax5_CCWlim} \n"
        ret += f"Axis 6: CW limit ({ax6_hasCWlim}): {ax6_CWlim}, CCW limit ({ax6_hasCCWlim}): {ax6_CCWlim}"

        print(ret)
        # return ret
    
        # limits = [(i, dir) for i in range(1, 7) for dir in ['CW', 'CCW']]
        # limit_values = {f'ax{i}_{dir}lim': self.get_lim(i, dir) if self.lim_enabled(i, dir) else None for i, dir in limits}
        # ret = '\n'.join([f'Axis {i}: CW limit: {limit_values[f"ax{i}_CWlim"]} ({self.lim_enabled(i, "CW")}), CCW limit: {limit_values[f"ax{i}_CCWlim"]} ({self.lim_enabled(i, "CCW")})' for i in range(1, 7)])
        # return ret


    def enable_all_lims(self) -> None:
        '''
        This function enables all limits on the mount.
        '''
        self.enable_lim(1, 'CW')
        self.enable_lim(1, 'CCW')
        self.enable_lim(2, 'CW')
        self.enable_lim(2, 'CCW')
        self.enable_lim(3, 'CW')
        self.enable_lim(3, 'CCW')
        self.enable_lim(4, 'CW')
        self.enable_lim(4, 'CCW')
        self.enable_lim(5, 'CW')
        self.enable_lim(5, 'CCW')
        self.enable_lim(6, 'CW')
        self.enable_lim(6, 'CCW')

    def disable_all_lims(self) -> None:
        '''
        This function disables all limits on the mount.
        '''
        self.disable_lim(1, 'CW')
        self.disable_lim(1, 'CCW')
        self.disable_lim(2, 'CW')
        self.disable_lim(2, 'CCW')
        self.disable_lim(3, 'CW')
        self.disable_lim(3, 'CCW')
        self.disable_lim(4, 'CW')
        self.disable_lim(4, 'CCW')
        self.disable_lim(5, 'CW')
        self.disable_lim(5, 'CCW')
        self.disable_lim(6, 'CW')
        self.disable_lim(6, 'CCW')
        

    def lim_enabled(self, axis:int, direction:str) -> bool:
        '''
        This function returns whether the soft limit is enabled in the specified direction.
        params: direction: str - "CW" or "CCW" where "CW" is the positive direction and "CCW" is the negative direction.
        returns: bool - True if limit is enabled, False if limit is disabled
        '''
        if direction == 'CW':
            cmd = f'AXI{axis}:CWSLE?'
        elif direction == 'CCW':
            cmd = f'AXI{axis}:CCWSLE?'
        else:
            raise ValueError('direction must be "CW" or "CCW"')
        
        ret = bool(int(self.read_command(cmd)))
        return ret

    def set_lim(self, axis:int, direction:str, limit: int) -> None:
        '''
        This function sets the upper limit of the soft limit.
        params: limit: int 
        params: direction: str - "CW" or "CCW" where "CW" is the positive direction and "CCW" is the negative direction.
        '''
        if direction == "CW":
            cmd = f'AXI{axis}:CWSLP {limit}'
        elif direction == "CCW":
            cmd = f'AXI{axis}:CCWSLP {limit}'
        else:
            raise ValueError('direction must be "CW" or "CCW"')
        
        self.send_command(cmd)

    
    def get_lim(self, axis:int, direction:str) -> int:
        '''
        This function returns the limit of the soft limit in the specified direction. 
        params: direction: str - "CW" or "CCW" where "CW" is the positive direction and "CCW" is the negative direction.
        returns: int - the soft limit
        '''
        if direction == "CW":
            cmd = f'AXI{axis}:CWSLP?'
        elif direction == "CCW":
            cmd = f'AXI{axis}:CCWSLP?'
        else:
            raise ValueError('direction must be "CW" or "CCW"')
    
        ret = int(self.read_command(cmd))
        return ret
    
    def set_unit(self, axis:int, unit: str) -> None:
        '''
        This function sets the units of the stage.
        params: unit: str - "pulse", "um", "mm", "deg", or "mrad".
        '''
        assert unit == "pulse" or unit == "mm" or unit == "um" or unit == "deg" or unit == "mrad", "unit must be 'pulse', 'um', 'mm', 'deg' or 'mrad'"

        if unit == "pulse":
            num = 0
        elif unit == "um":
            num = 1
        elif unit == "mm":
            num = 2
        elif unit == "deg":
            num = 3
        elif unit == "mrad":
            num = 4
        else:
            raise ValueError(f'unit must be "pulse", "um", "mm", "deg", or "mrad"')  # just incase assetion doesn't catch it

        cmd = f'AXI{axis}:UNIT {num}'
        self.send_command(cmd)


    def get_unit(self, axis:int) -> str:
        '''
        This function returns the units of the stage.
        returns: str - "pulse", "um", "mm", "deg", or "mrad".
        '''
        cmd = f'AXI{axis}:UNIT?'
        val = int(self.read_command(cmd))

        if val == 0:
            ret = "pulse"
        elif val ==1:
            ret == "um"
        elif val == 2:
            ret = "mm"
        elif val == 3:
            ret = "deg"
        elif val == 4:
            ret = "mrad"
        else:
            raise ValueError(f'returned unidentified unit {val}, must be 0, 1, 2, 3 or 4')
        return ret
    
    def in_motion(self, axis:int) -> bool:
        '''
        This function returns whether the stage is in motion.
        returns: bool - True if the stage is in motion, False if the stage is not in motion.
        '''
        cmd = f'AXI{axis}:MOTION?'
        ret = bool(int(self.read_command(cmd)))
        return ret

    def update_log(self, cmd:str):
        '''
        Anytime any of the above functions gets triggered, I want the log to be updated. there can be two logs, one that tracks EVERY command, and one that only tracks the position.
        '''

        dt = str(datetime.now())
        line = "{}: {}\n".format(dt, cmd)
        self.f.writelines(line)
        self.f.flush()


# if __name__ == "__main__":
#     mount = Mount('/dev/serial/by-id/usb-SURUGA_SEIKI_SURUGA_SEIKI_DS102-if00-port0', 38400)
#     # mount = Mount('/dev/ttyUSB0', 38400)
    
    



        