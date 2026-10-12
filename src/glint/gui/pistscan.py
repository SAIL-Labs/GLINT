from pyMilk.interfacing.shm import SHM
import numpy as np
from astropy.io import fits
import time
import tqdm
# from datetime import datetime
import csv
import glint_paths


# ------------------------------------------------------------
# Piston scan

# Tip and tilt values fo segments [11,20,31] in that order
# TIP = {11:-0.001, 20:-0.001, 31:-0.001}  #radians
# TILT = {11:-0.001, 20:-0.001, 31:-0.001}  #radians
TIP = {11:0.0, 20:0.0, 31:0.0}  #radians
TILT = {11:0.0, 20:0.0, 31:0.0}  #radians
SEGMENTS = [11,20,31]

def get_piston_range(segment):

    
    """
    Get the piston range for a segment.

    Parameters
    ----------
    segment : int
        Segment to get the piston range for.

    Returns
    -------
    minPiston : int
        Minimum piston value.
    maxPiston : int
        Maximum piston value.
    """

    if segment == 11:
        minPiston = -2500
        maxPiston = 400
    elif segment == 20:
        minPiston = -2600
        maxPiston = 250
    elif segment == 31:
        minPiston = -2650
        maxPiston = 200

    return minPiston, maxPiston
    


def dopistonscan(apapane, mems, baseline, tip, tilt, writer, stepsize) -> None:
    """
    Scans over two segments to find the null position. In this scan, teh number 1 refers to scanning over teh first segment, and teh number 2 refers to scanning over the second.

    Parameters
    ----------
    mems : apiMEMsControl.MEMS
        MEMs object.
    baseline : np.ndarray
        The two segments to scan over.
    tip : np.ndarray
        The tip values for the two segments.
    tilt : np.ndarray
        The tilt values for the two segments.
    n : int, optional
        Number of steps to take in the scan. The default is 100.

    Returns
    -------
    scan1 : np.ndarray
        Size (3xn): [nullscan, pos_seg1_scan1, pos_seg2_scan1]
        null scan is the summed flux at each step in the scan.
        pos_seg1_scan1 is the position of segment 1 at each step in the scan.
        pos_seg2_scan1 is the position of segment 2 at each step in the scan.
    scan2 : np.ndarray
        Size (3xn): [nullscan, pos_seg1_scan2, pos_seg2_scan2]
        null scan is the summed flux at each step in the scan.
        pos_seg1_scan2 is the position of segment 1 at each step in the scan.
        pos_seg2_scan2 is the position of segment 2 at each step in the scan.
    """
    
    seg1, seg2 = baseline
    tip_seg1, tip_seg2 = tip
    tilt_seg1, tilt_seg2 = tilt
    print(f'tip_seg1 = {tip_seg1}, tilt_seg1 = {tilt_seg1}')
    print(f'tip_seg2 = {tip_seg2}, tilt_seg2 = {tilt_seg2}')

    assert type(seg1) == int, 'Segment values must be integers'
    assert type(seg2) == int, 'Segment values must be integers'
    assert type(tip_seg1) == float, 'Tip values must be floats'
    assert type(tip_seg2) == float, 'Tip values must be floats'
    assert type(tilt_seg1) == float, 'Tilt values must be floats'
    assert type(tilt_seg2) == float, 'Tilt values must be floats'
    
    assert seg1 != seg2, 'Segments to scan over must be different'
    assert seg1 in SEGMENTS, 'Segment 1 must be one of the allowed segments: [11, 20, 31]'
    assert seg2 in SEGMENTS, 'Segment 2 must be one of the allowed segments: [11, 20, 31]'

    # Find the piston range for each segment
    minPiston_seg1, maxPiston_seg1 = get_piston_range(seg1) 
    print(f'Segment {seg1}: minpiston = {minPiston_seg1}, maxpiston = {maxPiston_seg1}')
    
    minPiston_seg2, maxPiston_seg2 = get_piston_range(seg2) 
    print(f'Segment {seg2}: minpiston = {minPiston_seg2}, maxpiston = {maxPiston_seg2}')

    scanrange = maxPiston_seg1 - minPiston_seg1

    # Number of steps to take in the scan
    n = int(scanrange/stepsize) + 1


    # Positions of the segments for each scan. The segment not being scanned will have constant values.
    # For the first scan, segment 2 wlil be at a minimum and sgement 1 will piston from maximum to minimum
    # For the second scan, segment 1 will be at a minimum and segment 2 will piston from minimum to maximum
    # Scan1 positions-----------------------------------------
    pos_seg1_scan1 = np.linspace(maxPiston_seg1, minPiston_seg1, n)

    # Scan2 positions-----------------------------------------
    pos_seg2_scan2 = np.linspace(minPiston_seg2, maxPiston_seg2, n)


    # Set the two segments to their maximum piston values to start.
    mems.set_segment(seg1, maxPiston_seg1, tip_seg1, tilt_seg1) 
    time.sleep(0.01)
    mems.set_segment(seg2, minPiston_seg2, tip_seg2, tilt_seg2)
    time.sleep(0.01)

    print(f'scanning {seg1} and {seg2}')

    pbar = tqdm.tqdm(desc="Piston scan", total=2*n)

    # 1. Loop over 2*n values.
    # 2. For the first n, keep segment 2 at its minimum piston value and scan segment 1 form maximum to minimum.
    # 3. Then at iteration n, segment 1 should be at a minimum position so move on to scanning over segment 2 from minimum to maximum
    for posnum in range (2*n):

        # Scan 1 -----------------------------------------
        if posnum < n:

            # New segment position
            newpos_seg1 = pos_seg1_scan1[posnum]

            # Get time
            # dt = str(datetime.now())
            t = time.time()
            
            mems.set_segment(seg1, newpos_seg1, tip_seg1, tilt_seg1)
            saveframe(apapane, baseline, newpos_seg1, t)


        # Scan 2 -----------------------------------------
        else:
            posnum = posnum - n  # Reset posnum to start from 0 again when scanning over segment 2
            
            # New segment position
            newpos_seg2 = pos_seg2_scan2[posnum]

            # Get time
            # dt = str(datetime.now())
            t = time.time()

            mems.set_segment(seg2, newpos_seg2, tip_seg2, tilt_seg2)
            saveframe(apapane, baseline, newpos_seg2, t)


        
        # Get the position
        pttpos = mems.get_ptt()
    
        # Write timestamp and positions to file
        writer.writerow(["Timestamp:", t])
        writer.writerow(["ptt:", pttpos])
        

        
        time.sleep(0.5)
        pbar.update()
    

    

    

def saveframe(apapane, baseline, pist, t):
    """
    Save a frame from the APAPANE camera.

    Parameters
    ----------
    apapane : SHM
        SHM object for the APAPANE camera.
    tip : float
        Tip position.
    tilt : float
        Tilt position.

    Returns
    -------
    None.

    """
    seg1, seg2 = baseline
    # Get the frame
    frame = apapane.multi_recv_data(10)
    frame = np.array(frame, dtype=float)  # Convert to float to avoid overflow

    # Save the frame
    hdu = fits.PrimaryHDU(frame)
    hdul = fits.HDUList([hdu])
    hdul.writeto(str(glint_paths.data_dir('gui', 'pistscans') / f'frame_baseline{seg1}:{seg2}_pist{pist}_{t}.fits'), overwrite=True)


def pistonscan(mems, cameras):
    # Variables
    # iteration = 16  # Scan iteration for saving reasons

    (apapane, _, _, _) = cameras

    for baseline in [[11,20], [11,31], [20,31]]:

        # Get the tip and tilt values for the segments
        seg1, seg2 = baseline
        tip = [TIP[seg1], TIP[seg2]]
        tilt = [TILT[seg1], TILT[seg2]]
        
         # Open a CSV file to log the data
        path = str(glint_paths.data_dir('gui', 'pistscans'))

        tstart = time.time()

        with open(path+f'baseline{seg1}:{seg2}_poslog_start{tstart}.csv', mode='x', newline='') as file:

            writer = csv.writer(file)

            # Perform the null scan
            dopistonscan(apapane, mems, baseline, tip, tilt, writer, stepsize = 30)