## Todo:
# - make the table editable and usable
# - enable functionality using baselines not just individual segments (done?)
# - enable function such that all segments except the usable ones move (done?)
# - make apapane view live
# - implement optimisation scans into gui
# - allow multi-frame saving from gui (done?)
# - implement tip/tilt in simulation mode



'''
This file provides funtionality to the GUI buttons. It is imported into the main GUI file, run_GUI.py. Where addditionalcomputation is required (e.g. null scan, chip simulator), the functions are called from this file.
'''
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import functions
# import sys
# sys.path.append('/home/scexao/glint/alignment_scans')
# import optimisationscansnight1
import pistscan
import ttscan
import numpy as np
from astropy.io import fits
import traceback
import json
import subprocess
from datetime import datetime, timezone
import ast

PREVIOUS_VALUES = {
    'tab_DMposition_base': {(row, col): 0 for row in range(3) for col in range(3)},
    'tab_DMposition_seg': {(row, col): 0 for row in range(37) for col in range(3)},
    'table_mountPos_xyz': {(row, col): 0 for row in range(1) for col in range(3)},
    'table_mountPos_rpy': {(row, col): 0 for row in range(1) for col in range(3)}
}

APERTURE_NUMBERS = [11, 20, 31]




def triggers(gui, dm, mount, cameras):

    ### Segment list ###
    # Replace the original focusInEvent with the new one
    gui.text_segments.focusInEvent = lambda event: focusInEvent_segments(gui)
    gui.text_segments.focusOutEvent = lambda event: test_segmentls_format(gui.text_segments.text(), gui.text_segments)


    ### Manual control checkboxes ###
    gui.checkBox_manualDM_base.stateChanged.connect(
        lambda state: toggleDMEditable(gui, state))
    gui.checkBox_manualDM_seg.stateChanged.connect(
        lambda state: toggleDMEditable(gui, state))
    gui.checkBox_manualMount.stateChanged.connect(
        lambda state: toggleMountEditable(gui, state))

    
    # We want this wrapped in a try-except block because tehre are various errors the mount can 
    # encounter and we don't want it to cause the entire GUI (and thus dm) to quit
    try:

        # gui.pushButton_TTscan.clicked.connect(
        #     lambda: ttscan.tiptiltscan(dm, cameras))
        # gui.pushButton_pistonscan.clicked.connect(
        #     lambda: pistscan.pistonscan(DM, cameras))
        

        ### MOUNT control buttons ###
        gui.pushButton_x_up.clicked.connect(
            lambda: move_mount(gui, mount, 'x', 'up'))
        gui.pushButton_x_down.clicked.connect(
            lambda: move_mount(gui, mount, 'x', 'down'))
        gui.pushButton_y_up.clicked.connect(
            lambda: move_mount(gui, mount, 'y', 'up'))
        gui.pushButton_y_down.clicked.connect(
            lambda: move_mount(gui, mount, 'y', 'down'))
        gui.pushButton_z_up.clicked.connect(
            lambda: move_mount(gui, mount, 'z', 'up'))
        gui.pushButton_z_down.clicked.connect(
            lambda: move_mount(gui, mount, 'z', 'down'))
        gui.pushButton_yaw_up.clicked.connect(
            lambda: move_mount(gui, mount, 'yaw', 'up'))
        gui.pushButton_yaw_down.clicked.connect(
            lambda: move_mount(gui, mount, 'yaw', 'down'))
        gui.pushButton_roll_up.clicked.connect(
            lambda: move_mount(gui, mount, 'roll', 'up'))
        gui.pushButton_roll_down.clicked.connect(
            lambda: move_mount(gui, mount, 'roll', 'down'))
        gui.pushButton_pitch_up.clicked.connect(
            lambda: move_mount(gui, mount, 'pitch', 'up'))
        gui.pushButton_pitch_down.clicked.connect(
            lambda: move_mount(gui, mount, 'pitch', 'down'))
        gui.pushButton_setLimit.clicked.connect(
            lambda: setLimit(gui, mount))
        gui.pushButton_getLimits.clicked.connect(
            lambda: getLimit(gui, mount))
        gui.pushButton_setSpeed.clicked.connect(
            lambda: setLimit(gui, mount))
        gui.pushButton_getSpeed.clicked.connect(
            lambda: getSpeed(gui, mount))
        gui.pushButton_setOrigin.clicked.connect(
            lambda: set_origin(gui, mount))
        gui.pushButton_getOrigin.clicked.connect(
            lambda: get_origin(gui, mount))
        gui.pushButton_origin.clicked.connect(
            lambda: origin(gui, mount))
        gui.pushButton_getpositions.clicked.connect(
            lambda: get_mountpos(gui, mount))
        gui.pushButton_stop.clicked.connect(
            lambda: stop(mount))

        gui.pushButton_scan_xy.clicked.connect(
            lambda: xyscan(gui))
        gui.pushButton_scan_pitchyaw.clicked.connect(
            lambda: pitchyawscan(gui))
        # gui.pushButton_scan_z.clicked.connect(
        #     lambda: zscan(gui))
        gui.pushButton_scan_tiptilt.clicked.connect(
            lambda: tiptiltscan(gui))
        gui.pushButton_scan_null.clicked.connect(
            lambda: nullscan(gui))
     
    
    
        ### dm control buttons ###
        gui.pushButton_piston_up_base.clicked.connect(
            lambda: move_dm(gui, dm, dmMode = 'base', command = 'piston', direction = 'up'))
        gui.pushButton_piston_down_base.clicked.connect(
            lambda: move_dm(gui, dm, dmMode = 'base', command = 'piston', direction = 'down'))
        gui.pushButton_tip_up_base.clicked.connect(
            lambda: move_dm(gui, dm, dmMode = 'base', command = 'tip', direction = 'up'))
        gui.pushButton_tip_down_base.clicked.connect(
            lambda: move_dm(gui, dm, dmMode = 'base', command = 'tip', direction = 'down'))
        gui.pushButton_tilt_up_base.clicked.connect(
            lambda: move_dm(gui, dm, dmMode = 'base', command = 'tilt', direction = 'up'))
        gui.pushButton_tilt_down_base.clicked.connect(
            lambda: move_dm(gui, dm, dmMode = 'base', command = 'tilt', direction = 'down'))
        gui.pushButton_piston_up_seg.clicked.connect(
            lambda: move_dm(gui, dm, dmMode = 'seg', command = 'piston', direction = 'up'))
        gui.pushButton_piston_down_seg.clicked.connect(
            lambda: move_dm(gui, dm, dmMode = 'seg', command = 'piston', direction = 'down'))
        gui.pushButton_tip_up_seg.clicked.connect(
            lambda: move_dm(gui, dm, dmMode = 'seg', command = 'tip', direction = 'up'))
        gui.pushButton_tip_down_seg.clicked.connect(
            lambda: move_dm(gui, dm, dmMode = 'seg', command = 'tip', direction = 'down'))
        gui.pushButton_tilt_up_seg.clicked.connect(
            lambda: move_dm(gui, dm, dmMode = 'seg', command = 'tilt', direction = 'up'))
        gui.pushButton_tilt_down_seg.clicked.connect(
            lambda: move_dm(gui, dm, dmMode = 'seg', command = 'tilt', direction = 'down'))
        
        gui.pushButton_flattenDM_base.clicked.connect(
            lambda: flatten(dm))
        gui.pushButton_flattenDM_seg.clicked.connect(
            lambda: flatten(dm))
            
        gui.pushButton_zeroDMvolts_seg.clicked.connect(
            lambda: set_zerovolts(dm))
        gui.pushButton_zeroDMvolts_base.clicked.connect(
            lambda: set_zerovolts(dm))
        
        gui.pushButton_zeroDMptt_seg.clicked.connect(
            lambda: set_zeroptt(gui, dm))
        gui.pushButton_zeroDMptt_base.clicked.connect(
            lambda: set_zeroptt(gui, dm))
        
        
        
        gui.pushButton_save.clicked.connect(
            lambda: save_frames(gui, cameras))
        
        
        ### Spectral Simulator ###
        # gui.checkBox_simMode.stateChanged.connect(
        #     lambda state: toggleLiveCam(gui, state))
        
        
        ### Set step sizes ###
        gui.text_pistStepSize_base.textChanged.connect(
            lambda text, le=gui.text_pistStepSize_base: on_text_changed(text, le))
        gui.text_ttStepSize_base.textChanged.connect(
            lambda text, le=gui.text_ttStepSize_base: on_text_changed(text, le))
        
        gui.text_mountStepSize_urad.textChanged.connect(
            lambda text, le=gui.text_mountStepSize_urad: on_text_changed(text, le))
        gui.text_mountStepSize_um.textChanged.connect(
            lambda text, le=gui.text_mountStepSize_um: on_text_changed(text, le))


        gui.tab_DMposition_base.itemChanged.connect(
            lambda item, device=dm, table=gui.tab_DMposition_base, linked_table=gui.tab_DMposition_seg: 
            on_item_changed(item, device, table, linked_table)
        )
        gui.tab_DMposition_seg.itemChanged.connect(
            lambda item, device=dm, table=gui.tab_DMposition_seg, linked_table=gui.tab_DMposition_base: 
            on_item_changed(item, device, table, linked_table)
        )
    
        gui.table_mountPos_rpy.itemChanged.connect(
            lambda item, device = mount, table=gui.table_mountPos_rpy, linked_table=None: 
            on_item_changed(item, device, table,linked_table)
            )
        gui.table_mountPos_xyz.itemChanged.connect(
            lambda item, device = mount, table=gui.table_mountPos_xyz, linked_table=None: 
            on_item_changed(item, device, table,linked_table)
            )

        
        
    
    except Exception as e:
        print("Error:", e)

### GENERAL FUNCTIONS ###

def update_cell(table, r, col, value):

    # Delegate ensures that when a user edits the cell, the input is formatted and displayed as an integer without decimals.
    item = QtWidgets.QTableWidgetItem(str(value))
    table.blockSignals(True)
    table.setItem(r, col, item)
    table.blockSignals(False)

def get_checked_states(gui):
    # Get the checked state of each item in the QListWidget
    checked_states = []

    for i in range(gui.list_segments.count()):
        item = gui.list_segments.item(i)
        if item.checkState() == Qt.Checked:
            checked_states.append(int(item.text()))

    return checked_states

def toggleLiveCam(gui, state):
    if state == 2:  # 2 corresponds to checked state
        gui.widget_selectFrame.setEnabled(False)
    else:
        gui.widget_selectFrame.setEnabled(True)

def on_text_changed(text, variable):

    try:
        value = float(text)
    except ValueError:
        variable.setText(None)
        print("Input value is not a float.")


def map_row_to_compressed(row):
    """Map a full table row to a compressed table row."""
    mapping = {11: 0, 20: 1, 31: 2}
    return mapping.get(row, None)

def map_row_to_full(row_compressed):
    """Map a compressed table row back to a full table row."""
    mapping = {0: 11, 1: 20, 2: 31}
    return mapping.get(row_compressed, None)



def update_linked_table(linked_table, row, col, value):
    if linked_table.rowCount() < 10:  # assume compressed table
        mapped_row = map_row_to_compressed(row)
    else:
        mapped_row = map_row_to_full(row)

    if mapped_row is not None:
        linked_item = linked_table.item(mapped_row, col)
        if linked_item is not None:
            linked_table.blockSignals(True)
            linked_item.setText("{:.2f}".format(value))
            linked_table.blockSignals(False)



def on_item_changed(item, device, table, linked_table):
    text = table.item(item.row(), item.column()).text()
    row = item.row()
    col = item.column()
    table_name = table.objectName()

    try:
        new_value = float(text)

        if device is not None:
            move(device, table, row, col, new_value)

        PREVIOUS_VALUES[table_name][(row, col)] = new_value

        table.blockSignals(True)
        table.item(row, col).setText("{:.2f}".format(new_value))
        table.blockSignals(False)

        if linked_table is not None:
            update_linked_table(linked_table, row, col, new_value)

    except ValueError:
        prev_value = PREVIOUS_VALUES[table_name].get((row, col), 0)

        table.blockSignals(True)
        table.item(row, col).setText("{:.2f}".format(prev_value))
        table.blockSignals(False)


# def on_item_changed(item, device, table):
#     '''    
#     psudeocode:
#         1. get the new value
#         2. if the value is not a float, prevent the change
#         3. else, continue
#         5. send the command
#         6. get the new position
#         7. update the cell
#     '''
#     text = item.text()
#     row = item.row()
#     col = item.column()
#     table_name = table.objectName()
    

#     try:
#         float_value = float(text)
#         new_value = float_value

#         if device is not None:
#             move(device, table, row, col, new_value)

#         # update previous value
#         PREVIOUS_VALUES[table_name][(row, col)] = new_value

#         # Prevent the cellChanged signal from being triggered otherwise previous value will be overwritten
#         table.blockSignals(True)
#         item.setText("{:.3f}".format(new_value))  # Convert to 3 decimal places if it's a float
#         table.blockSignals(False)

#     except ValueError: # If the entered value is not a float, prevent the change
#         previous_table = PREVIOUS_VALUES[table_name]
#         prev_value = previous_table.get((row, col), 0)

#         table.blockSignals(True)
#         item.setText("{:.3f}".format(prev_value))
#         table.blockSignals(False)



def check_stepSize(stepsize):
   
    try:
        step = stepsize.text()
        step = float(step)
        return step
    except ValueError:
        return None

def move(device, table, row, col, value):

    '''
    pseudocode:
    mems:
        1. find the segment
        2. check the limits
        3. set the segment
        4. get the new position
    mount:
        1. get axis (p = 1, r = 2, y = 3, x = 4, z = 5, y = 6)
        2. check the limits
        3. set the position
        4. get the new position
    '''
    
    if table.objectName() == 'tab_DMposition_seg':
        
        segment = row
        piston = float(table.item(row, 0).text())
        tip = float(table.item(row, 1).text())
        tilt = float(table.item(row, 2).text())

        device.set_segment(segment, piston, tip, tilt)

    elif table.objectName() == 'tab_DMposition_base':
        segment = APERTURE_NUMBERS[row]

        piston = float(table.item(row, 0).text())
        tip = float(table.item(row, 1).text())
        tilt = float(table.item(row, 2).text())

        device.set_segment(segment, piston, tip, tilt)
        
    elif table.objectName() == 'table_mountPos_xyz':
        if col == 0:
            axis = 4
        elif col == 1:
            axis = 6
        elif col == 2:
            axis = 5
        
        value  = int(value)
        
        device.set_pos(axis, value)
    
    elif table.objectName() == 'table_mountPos_rpy':
        if col == 0:
            axis = 2
        elif col == 1:
            axis = 1
        elif col == 2:
            axis = 3

        value  = int(value)
        
        device.set_pos(axis, value)

    


### MOUNT FUNCTIONS ###


# set lim
def setLimit(gui, mount):
    if mount is None:
        print("No mount connected.")
        return
    

    if gui.radioButton_CW.isChecked():
        direction = "CW"
    elif gui.radioButton_CCW.isChecked():
        direction = "CCW"
    else:
        print("Select direction CW or CCW.")
        return
    
    try:
        axis = int(gui.text_axis.text())
    except ValueError:
        print("Invalid axis number: 1 (pitch), 2 (roll), 3 (yaw), 4 (x), 5 (z), 6 (y).")
        return

    try:
        limit = int(gui.text_setLimit.text())
    except ValueError:
        print("Invalid limit: must be an integer.")
        return

    mount.enable_lim(axis, direction)
    mount.set_lim(axis, direction, limit)


def getLimit(gui, mount):
    if mount is None:
        print("No mount connected.")
        return

    try:
        axis = int(gui.text_axis.text())
    except ValueError:
        print("Invalid axis number: 1 (pitch), 2 (roll), 3 (yaw), 4 (x), 5 (z), 6 (y).")
        return
    
    if gui.radioButton_CW.isChecked():
        direction = "CW"
    elif gui.radioButton_CCW.isChecked():
        direction = "CCW"
    else:
        print("Select direction CW or CCW.")
        return
    
    try:
        limit = mount.get_lim(axis, direction)
        gui.label_getLimits.setText(str(limit))
        
    except Exception as e:
        print("Error getting speed:", e)
        return



def toggleMountEditable(gui, state):
    if state == 2:  # 2 corresponds to checked state
        gui.widget_manualMount_inner.setEnabled(True)
        gui.table_mountPos_rpy.setEditTriggers(QtWidgets.QTableWidget.AllEditTriggers)
        gui.table_mountPos_xyz.setEditTriggers(QtWidgets.QTableWidget.AllEditTriggers)
    else:
        gui.widget_manualMount_inner.setEnabled(False)
        gui.table_mountPos_rpy.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        gui.table_mountPos_xyz.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)

def setSpeed(gui, mount):
    if mount is None:
        print("No mount connected.")
        return
        

    try:
        speed = int(gui.text_setSpeed.text())
    except ValueError:
        print("Invalid speed: must be an integer between 1 (slowest) and 6 (fastest).")
        return
    
    try:
        axis = int(gui.text_axis.text())
    except ValueError:
        print("Invalid axis number: 1 (pitch), 2 (roll), 3 (yaw), 4 (x), 5 (z), 6 (y).")
        return
    
    try:
        mount.set_speed(axis, speed)

    except Exception as e:
        print("Error setting speed:", e)
        return

def getSpeed(gui, mount):

    if mount is None:
        print("No mount connected.")
        return

    try:
        axis = int(gui.text_axis.text())
    except ValueError:
        print("Invalid axis number: 1 (pitch), 2 (roll), 3 (yaw), 4 (x), 5 (z), 6 (y).")
        return
    
    try:
        speed = mount.get_speed(axis)
        gui.label_getSpeed.setText(str(speed))
        
    except Exception as e:
        print("Error getting speed:", e)
        return

def set_origin(gui, mount):

    print('Disabled origin function.')

    # if mount is None:
    #     print("No mount connected.")
    #     return

    # try:
    #     axis = int(gui.text_axis.text())
    # except ValueError:
    #     print("Invalid axis number: 1 (pitch), 2 (roll), 3 (yaw), 4 (x), 5 (z), 6 (y).")
    #     return
    
    # try:
    #     pattern = int(gui.text_setOrigin.text())
    # except ValueError:
    #     print("Invalid origin pattern: 3 (clockwise), 4 (counter-clockwise).")
    #     return
    
    # try:
    #     mount.set_origin_pattern(axis, pattern)

    # except Exception as e:
    #     print("Error setting origin:", e)
    #     return

def get_origin(gui, mount):

    print("Disabled origin function.")
    # if mount is None:
    #     print("No mount connected.")
    #     return

    # try:
    #     axis = int(gui.text_axis.text())
    # except ValueError:
    #     print("Invalid axis number: 1 (pitch), 2 (roll), 3 (yaw), 4 (x), 5 (z), 6 (y).")
    #     return
    
    # try:
    #     pattern = mount.get_origin_pattern(axis)
    #     gui.label_getOrigin.setText(str(pattern))
    # except Exception as e:
    #     print("Error getting origin pattern:", e)
    #     return



def preorigin_checks():

    '''
    check the following:
    1) origin pattern is correct
    2) position relative to origin is good to proceed with origining
    3) if not, a pop-up message should appear
    '''
    pass



def origin(gui, mount):

    print('Disabled origin function.')
    # if mount is None:
    #     print("No mount connected.")
    #     return

    # try:
    #     axis = int(gui.text_axis.text())
    # except ValueError:
    #     print("Invalid axis number: 1 (pitch), 2 (roll), 3 (yaw), 4 (x), 5 (z), 6 (y).")
    #     return
    
    # try:
    #     mount.go_origin(axis)
        
    # except Exception as e:
    #     print("Error going to origin:", e)
    #     return



def stop(mount):

    if mount is None:
        print("No mount connected.")
        return
    
    try:
        mount.rstop()
    except Exception as e:
        print("Error stopping mount:", e)
        return


'''
Pseudocode:
Get the current mount position
Add the step size to the current position
Set the new position
'''
def unpack_mounttable(gui, command):

    if command == 'x':
        table = gui.table_mountPos_xyz
        row = 0
        col = 0
        axis = 4
        stepsize = check_stepSize(gui.text_mountStepSize_um) # get stepsize and check if it's a float
    elif command == 'y':
        table = gui.table_mountPos_xyz
        row = 0
        col = 1
        axis = 6
        stepsize = check_stepSize(gui.text_mountStepSize_um)
    elif command == 'z':
        table = gui.table_mountPos_xyz
        row = 0
        col = 2
        axis = 5
        stepsize = check_stepSize(gui.text_mountStepSize_um)
    elif command == 'roll':
        table = gui.table_mountPos_rpy
        row = 0
        col = 0
        axis = 2
        stepsize = check_stepSize(gui.text_mountStepSize_urad)
    elif command == 'pitch':
        table = gui.table_mountPos_rpy
        row = 0
        col = 1
        axis = 1
        stepsize = check_stepSize(gui.text_mountStepSize_urad)
    elif command == 'yaw':
        table = gui.table_mountPos_rpy
        row = 0
        col = 2
        axis = 3
        stepsize = check_stepSize(gui.text_mountStepSize_urad)
    else:
        print("Invalid command.")
        return

    return (table, row, col, axis, stepsize)


def move_mount(gui, mount, command, direction):

    # Get the table and the row and column of the cell to be updated
    (table, row, col, axis, stepsize) = unpack_mounttable(gui, command)

    if stepsize is not None:

        ### MAYBE CHANGE THIS TO IF SIMULATION CHECKBOX IS TICKED
        if mount is not None:   # i.e. if in simulation mode and can't connect to the mount
        #     # Get the current position from the table
        #     pos = float(table.item(row, col).text())

        #     # Add the step size to the current position

        #     if direction == 'up':
        #         updated_pos = pos + stepsize
        #     elif direction == 'down':
        #         updated_pos = pos - stepsize
        #     else:
        #         print("Invalid direction.")
        #         return
            
        # else:

            # Get the current position of the stage
            pos = mount.get_pos(axis)

            # Add the step size to the current position
            if direction == 'up':
                newpos = int(pos + stepsize)
            elif direction == 'down':
                newpos = int(pos - stepsize)
            else:
                print("Invalid direction.")
                return

            # Set the new position
            mount.set_pos(axis, newpos)
    else:
        print("Mount step size not given.")

def get_mountpos(gui, mount):

    if mount is None:
        print("No mount connected.")
        return
    
    axes = ['x', 'y', 'z', 'pitch', 'roll', 'yaw']

    for ax in axes:
        (table, row, col, axis, _) = unpack_mounttable(gui, ax)
        updated_pos = mount.get_pos(axis)

        update_cell(table, row, col, updated_pos)  

# import ast

def parseval(value):
    # Try Python literal style, e.g. "[3, 5]" or "(3, 5)"
    try:
        parsed = ast.literal_eval(value)
        if (
            isinstance(parsed, (list, tuple)) and
            len(parsed) == 2 and
            all(isinstance(i, (int, float)) for i in parsed)
        ):
            return [float(i) for i in parsed]  # ensure consistent float output
    except Exception:
        pass

    # Try comma-separated style, e.g. "3,5" or "0.15, -0.2"
    try:
        parts = [float(p.strip()) for p in value.split(',')]
        if len(parts) == 2:
            return parts
    except Exception:
        pass

    print(f"Invalid input format. Expected [a, b] where a and b are numbers. Got: {value!r}")
    return None

def xyscan(gui):
    '''
    pseudocode:
    1. access alignment_scans folder
    2. get the text in the gui
    3. set the text in the .json file
    4. Set the date in json file

    LATER:
    1. automatically identify the boxes for the photometric channels
    2. allow the user to select which photometric channels they want to save a scan for (though there's no harm in doing all of them)
    '''    
    # Load settings
    
    scanparams_path = "/home/scexao/glint/alignment_scans/xyscans/scanparameters.json"
    with open(scanparams_path, 'r') as f:
        params = json.load(f)

    step_size = parseval(gui.text_stepsize_xy.text())
    start_pos = parseval(gui.text_startpos_xy.text())
    scan_range = parseval(gui.text_range_xy.text())

    if None in (step_size, start_pos, scan_range):
        return

    # Get the current UTC year and date
    # Get current UTC time
    now = datetime.now(timezone.utc)
    params['year'] = now.year
    params['date'] = now.strftime("%m-%d")  # e.g., "10-29"
    params['step_size'] = step_size
    params['scan_range'] = scan_range
    params['start_pos'] = start_pos

    # Save updated parameters
    with open(scanparams_path, 'w') as f:
        json.dump(params, f, indent=4)

    # Run the external script
    subprocess.run(["python3", "/home/scexao/glint/alignment_scans/xyscans/xyscan.py"])

    # Run the external script
    # subprocess.run(["python3", "/home/scexao/glint/alignment_scans/xyscans/read_xyscan.py"])


def pitchyawscan(gui):
    '''
    pseudocode:
    1. access alignment_scans folder
    2. get the text in the gui
    3. set the text in the .json file
    4. Set the date in json file

    LATER:
    1. automatically identify the boxes for the photometric channels
    2. allow the user to select which photometric channels they want to save a scan for (though there's no harm in doing all of them)
    '''    
    # Load settings
    
    scanparams_path = "/home/scexao/glint/alignment_scans/pitchyawscans/scanparameters.json"
    with open(scanparams_path, 'r') as f:
        params = json.load(f)

    step_size = parseval(gui.text_stepsize_pitchyaw.text())
    start_pos = parseval(gui.text_startpos_pitchyaw.text())
    scan_range = parseval(gui.text_range_pitchyaw.text())

    if None in (step_size, start_pos, scan_range):
        return

    # Get the current UTC year and date
    # Get current UTC time
    now = datetime.now(timezone.utc)
    params['year'] = now.year
    params['date'] = now.strftime("%m-%d")  # e.g., "10-29"
    params['step_size'] = step_size
    params['scan_range'] = scan_range
    params['start_pos'] = start_pos

    # Save updated parameters
    with open(scanparams_path, 'w') as f:
        json.dump(params, f, indent=4)

    # Run the external script
    subprocess.run(["python3", "/home/scexao/glint/alignment_scans/pitchyawscans/pitchyawscan.py"])


def tiptiltscan(gui):
    '''
    pseudocode:
    1. access alignment_scans folder
    2. get the text in the gui
    3. set the text in the .json file

    LATER:
    1. automatically identify the boxes for the photometric channels
    2. allow the user to select which photometric channels they want to save a scan for (though there's no harm in doing all of them)
    '''    
    # Load settings
    
    scanparams_path = "/home/scexao/glint/alignment_scans/tiptiltscans/scanparameters.json"
    with open(scanparams_path, 'r') as f:
        params = json.load(f)

    step_size = parseval(gui.text_stepsize_tiptilt.text())
    start_pos = parseval(gui.text_startpos_tiptilt.text())
    scan_range = parseval(gui.text_range_tiptilt.text())

    if None in (step_size, start_pos, scan_range):
        return

    params['step_size'] = step_size
    params['scan_range'] = scan_range
    params['start_pos'] = start_pos

    # Save updated parameters
    with open(scanparams_path, 'w') as f:
        json.dump(params, f, indent=4)

    # Run the external script
    subprocess.run(["python3", "/home/scexao/glint/alignment_scans/tiptiltscans/tiptiltscan.py"])

def parse_float(value, name="value"):
    """
    Safely parse a single float.
    Returns float or None if invalid.
    """
    try:
        value = value.strip()
        if value == "":
            raise ValueError("Empty input")

        return float(value)

    except Exception:
        print(f"Invalid input for {name}: {value!r}")
        return None
    

def nullscan(gui):

    scanparams_path = "/home/scexao/glint/alignment_scans/nullscans/scanparameters.json"

    with open(scanparams_path, 'r') as f:
        params = json.load(f)

    # Safely parse inputs
    step_size = parse_float(gui.text_stepsize_null.text(), "step_size")
    start_pos = parse_float(gui.text_startpos_null.text(), "start_pos")
    scan_range = parse_float(gui.text_range_null.text(), "scan_range")

    # Stop if any input is invalid
    if None in (step_size, start_pos, scan_range):
        print("Aborting nullscan due to invalid input.")
        return

    # Update params
    params['step_size'] = round(step_size, 3)
    params['scan_range'] = round(scan_range, 3)
    params['start_pos'] = round(start_pos, 3)

    # Save updated parameters
    with open(scanparams_path, 'w') as f:
        json.dump(params, f, indent=4)

    # Run the external script
    subprocess.run(["python3", "/home/scexao/glint/alignment_scans/nullscans/nullscan.py"])


### dm FUNCTIONS ###

def flatten(dm):

    ## IMPORTANT: This does not send the ptt to zero, so the table technically shouldn't be zero.
    # Check is connected to dm, in which case, flatten it
    if dm is not None: 
        dm.flatten()

    else:
        print("No dm connected.")
    
    
def set_zerovolts(dm):
    ## IMPORTANT: This does not send the ptt to zero, just the volts, so the table technically shouldn't be zero.
    # Check is connected to dm, in which case, send zeros
    if dm is not None: 
        dm.sendzeros()


    else:
        print("No dm connected.")


def set_zeroptt(gui, dm):

    if dm is not None: 
        dm.set_ptt_all(0,0,0)
    else:
        print("No dm connected.")
    
    for r in range(37):
        for c in range(3):

            update_cell(gui.tab_DMposition_base, r, c, 0)
            update_cell(gui.tab_DMposition_seg, r, c, 0)


def toggleList(gui):
    # Toggle the visibility of the list widget
    gui.list_segments.setVisible(not gui.list_segments.isVisible())
    # gui.list_segments.raise_()

# def setTableEditable(table, editable):
#     if editable:
#         table.setEditTriggers(QtWidgets.QTableWidget.AllEditTriggers)
#     else:
#         table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
     
def toggleDMEditable(gui, state):
    '''
    This could be generalised but my brain hurts
    '''

    if state == 2:  # 2 corresponds to checked state
        gui.widget_manualDM_inner_base.setEnabled(True)
        gui.widget_manualDM_inner_seg.setEnabled(True)
        gui.tab_DMposition_seg.setEditTriggers(QtWidgets.QTableWidget.AllEditTriggers)
        gui.tab_DMposition_base.setEditTriggers(QtWidgets.QTableWidget.AllEditTriggers)
    else:
        gui.widget_manualDM_inner_base.setEnabled(False)
        gui.widget_manualDM_inner_seg.setEnabled(False)
        gui.tab_DMposition_base.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        gui.tab_DMposition_seg.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)


def test_segmentls_format(text, le):

    '''
    Copilot wrote this code
    '''
    default = "E.g. 0, 4"

    if not text:
        le.setText(default)
    else:
        try:
            # Check if the string is in the format of an array
            segments = [int(s.strip()) for s in text.split(',')]
            print(segments)
            # If the format is correct, set the text to the input text
            le.setText(text)
        except ValueError:
            le.setText(default)
            print("Invalid format. Please enter numbers separated by commas.")

# Modify the focusInEvent to not clear the text if it's in the correct format
def focusInEvent_segments(gui):

    segments = get_segments(gui)

    if segments is None:
        gui.text_segments.setText('')

def focusInEvent_dm(gui):
    pass

    
def get_segments(gui):

    try:
        text_segments = gui.text_segments.text()
        segments = [int(s.strip()) for s in text_segments.split(',')]
        return segments
    except ValueError:
        print("No segment given.")
        return None
        
def get_baselines(gui):

    baseline = gui.comboBox_baselines.currentText()

    # Convert the baseline array into a list of integers
    try:
        baseline_list = [int(s.strip()) for s in baseline.split(',')]
    
    except ValueError:
        print("Invalid baseline format.")
        return

    if gui.radioButton_seg1.isChecked():
        aperture = baseline_list[0]
    elif gui.radioButton_seg2.isChecked():
        aperture = baseline_list[1]
    else:
        print('No segment selected.')
        return

    segments = [aperture] # Subtract 1 to get the correct index, put it in a list to match the format of the other functions

    return segments

def unpack_dm(gui, dmMode, command):

    if dmMode == 'base':
        if command == 'piston':
            stepsize = check_stepSize(gui.text_pistStepSize_base)
            col = 0
        elif command == 'tip':
            stepsize = check_stepSize(gui.text_ttStepSize_base)
            col = 1
        elif command == 'tilt':
            stepsize = check_stepSize(gui.text_ttStepSize_base)
            col = 2
        else:
            print('Invalid command.')
            return
    elif dmMode == 'seg':
        if command == 'piston':
            stepsize = check_stepSize(gui.text_pistStepSize_seg)
            col = 0
        elif command == 'tip':
            stepsize = check_stepSize(gui.text_ttStepSize_seg)
            col = 1
        elif command == 'tilt':
            stepsize = check_stepSize(gui.text_ttStepSize_seg)
            col = 2
        else:
            print('Invalid command.')
            return

    return (stepsize, col)



def move_dm(gui, dm, dmMode, command, direction): # dm mode refers to if you use segments or baselines

    # Initialise the table, stepsize, and segments based on the dm mode i.e. 'base' (uses a selected baseline) or 'seg' (uses segment numbers)
    if dmMode == 'base':
        segments = get_baselines(gui)
    elif dmMode == 'seg':
        segments = get_segments(gui)
    else:
        print('Invalid dm mode.')
        return
    
    if segments is None: 
        # If no segments are given, return
        return
    
    # Main table used to get pistons from and update pistons to
    main_table = gui.tab_DMposition_seg
    (stepsize, col) = unpack_dm(gui, dmMode, command)

    # Make sure the stepsize exists
    if stepsize is not None:

        # 37 will select all segments, -1 will select all segments except usable ones
        if 37 in segments:
            segments = list(range(37))
        elif -1 in segments:
            segments = list(range(37))

            # remove the relevant segments
            for aperturenum in APERTURE_NUMBERS:
                segments.remove(aperturenum)


        # Go through each segment selected to change the piston value
        for segment in segments:
            if segment < 0 or segment > 36:
                print('Incorrect segment number')
                return

            # Get the old piston value
            oldvalue = float(main_table.item(segment, col).text())
            # oldvalue = dm.get_ptt(

            # Calculate the new piston value 
            if direction == 'up':
                newvalue = oldvalue + stepsize
            elif direction == 'down':
                newvalue = oldvalue - stepsize

            # Update both tables with the new piston value
            update_cell(main_table, segment, col, newvalue)
            
            
            
            # Send the new piston value to the dm
            if not gui.checkBox_simMode.isChecked() and dm is not None:  # if not in simulation mode and we have properly opened the dm, move a segment
                
                pist = float(main_table.item(segment, 0).text())
                tip = float(main_table.item(segment, 1).text())
                tilt = float(main_table.item(segment, 2).text())

                # tiprad = tip/1000
                # tiltrad = tilt/1000

                try:
                    error = dm.set_segment(segment, pist, tip, tilt) 

                    if error == -1:
                        print("No update to table.")
                        return
                except Exception:
                    traceback.print_exc()
                    return
                
            
            

            # Find the index of the segment in the table i.e. instead of 1,3,9 it would be 0,1,2
            if segment in APERTURE_NUMBERS:
                s = APERTURE_NUMBERS.index(segment)
                update_cell(gui.tab_DMposition_base, s, col, newvalue)
            
            
            
        # Update the spectra if in simulation mode
        if gui.checkBox_simMode.isChecked():
            pistCol = 0
            pistons = np.array([float(main_table.item(segment, pistCol).text()) for segment in APERTURE_NUMBERS])
            functions.update_spectra(gui, pistons = pistons)
        
    else:
        print("dm step size not given.")



from datetime import datetime
def save_frames(gui, cameras):

    path = gui.text_path.text()
    
    checkboxes = [gui.checkBox_apapane, gui.checkBox_palila, gui.checkBox_pupil, gui.checkBox_focal]
    camnames = ['apapane', 'palila', 'glint_pupil', 'glint_focal']

    for idx, box in enumerate(checkboxes):
        if box.isChecked():

            if cameras is None:  # Are the cameras initialised or is this running locally
                frame = np.zeros((4,4))
            elif camnames[idx] == 'apapane':  # Apapane save data works slightly different to accomodate for multiple saves
                
                try: 
                    numframes = int(gui.text_numframes.text())
                    frame = cameras[idx].multi_recv_data(numframes)
                except ValueError:
                    numframes = 1 # force at least one frame to be saved
                    gui.text_numframes.setText(str(numframes))
                    print('Number of apapane frames to save must be an integer.')
                    return
            
            else:  # Save a single frame
                frame = cameras[idx].get_data()      
            
            dt = str(datetime.now())
            dt = dt.replace(' ', '_')

            if gui.text_filename.text() == "":  # i.e. if no filename was given in the gui, make a filename that saves it as a timestamp
                filename = camnames[idx] + '_' + dt 
            else:
                filename = gui.text_filename.text()

            try:
                # Save data as a fits file
                hdu = fits.PrimaryHDU(frame)
                hdul = fits.HDUList([hdu])
                hdul.writeto(path +'/'+ filename + '.fits', overwrite=False)
            except Exception as e:
                print('Save error occurred:', e)
        
