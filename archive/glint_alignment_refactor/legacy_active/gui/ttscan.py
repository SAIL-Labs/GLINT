import sys
sys.path.append('/home/scexao/glint/control-code/')

from pyMilk.interfacing.shm import SHM
import numpy as np
from astropy.io import fits
import time
import tqdm
# from datetime import datetime
import csv


 



# ------------------------------------------------------------

## TT scan

def tiptiltscan(mems, cameras):

    (apapane, _, _, _) = cameras

    takedark = False
    if takedark:
        dark_filepath = '/home/scexao/glint/observing/night2/dark.fits'
        dark = getdark(dark_filepath)
    else:
        dark = None
    
    # Set up the scan
    scan_range = np.array([0.01, 0.01])
    step_size = np.array([0.001, 0.001])
    start_tip = -0.005
    start_tilt = -0.005
    start_pos = [start_tip, start_tilt]

    # Apapane spectral boxes
    peaks = [237,218,199]  # Centre of the spectral boxes
    boundingvals = [200,260]  # Bounding the length of the spectral boxes
    box_halfwidth = 2  # Halfwidth of the spectral box to sum over
    iscred1 = True   # If True, the spectra are horizontal, if False, the spectra are vertical 

    # Get the spectral boxes: [top, bottom, left, right]
    boxes = [getbox(peak, boundingvals, box_halfwidth, iscred1) for peak in peaks]  

    for segment in [11, 20, 31]:
        # Spectra crop
        if segment == 11:
            box = boxes[0]
            piston = -1200
        elif segment == 20:
            box = boxes[1]
            piston = -1200
        elif segment == 31:
            box = boxes[2]
            piston = -1200
        else:
            print('Invalid segment')

        # Open a CSV file to log the data
        path = '/home/scexao/glint/observing/night2/ttscans/'
        # dtstart = str(datetime.now())
        # dtstart = dtstart.replace(' ', '_')

        tstart = time.time()


        

        with open(path+f'seg{segment}poslog_start{tstart}.csv', mode='x', newline='') as file:
            writer = csv.writer(file)

            # Get the segment positions
            nsteps = np.ceil(scan_range/step_size).astype(int) + 1  # Number of steps in the scan (need to plus 1 to include the last position)
            segpositions = [np.linspace(start_pos[i], start_pos[i] + scan_range[i], nsteps[i]) for i in range(2)]
            segpositions = np.array(segpositions)
        
            scan = doTTscan(mems, segment, segpositions, piston, apapane, dark, box, writer)


        # Save data
        hdu = fits.PrimaryHDU(scan)
        hdul = fits.HDUList([hdu])
        hdul.writeto(f'/home/scexao/glint/observing/night2/ttscans/scans/tiptilt_seg{segment}_start{tstart}.fits', overwrite=True)

def saveframe(apapane, segment, tip, tilt, t):
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

    # Get the frame
    frame = apapane.multi_recv_data(10)
    frame = np.array(frame, dtype=float)  # Convert to float to avoid overflow

    # Save the frame
    hdu = fits.PrimaryHDU(frame)
    hdul = fits.HDUList([hdu])
    hdul.writeto(f'/home/scexao/glint/observing/night2/ttscans/frame_seg{segment}_tip{tip}_tilt{tilt}_{t}.fits', overwrite=True)



def getbox(peak, boundingvals,  box_halfwidth, iscred1):
    """
    Get the spectral box for the scan.

    Parameters
    ----------
    peak : int
        Centre of the spectral box.
    boundingvals : list
        The bounding values of the spectral box.
    box_halfwidth : int
        Halfwidth of the spectral box to sum over.
    iscred1 : bool

    Returns
    -------
    box : list
        [top, bottom, left, right]
        Top of the spectral box etc.
    """

    val1, val2 = boundingvals

    # If True, the spectra are horizontal, if False, the spectra are vertical
    if iscred1:
        top = peak-box_halfwidth
        bottom = peak+box_halfwidth
        left = val1
        right = val2
    else:
        left = peak-box_halfwidth
        right = peak+box_halfwidth
        top = val1
        bottom = val2

    box = [top, bottom, left, right]

    return box




def getdark(dark_filepath: str) -> np.ndarray:
    """
    Get the dark frame.

    Parameters
    ----------
    dark_filepath : str
        Filepath to the dark frame.

    Returns
    -------
    dark : np.ndarray
        Dark frame.

    """
    with fits.open(dark_filepath) as hdul:
        dark = hdul[0].data[0, 3:-3, 3:-3]  # This crop is to remove the magic pixel
        dark = np.array(dark, dtype=float)  # Convert to float to avoid overflow

    return dark



def getdata(apapane, box, dark, nframes = 1) -> np.ndarray:
    """
    Takes data from the APAPANE camera, subtracts the dark frame, and crops the data to the spectral box.

    Parameters
    ----------
    apapane : SHM
        SHM object for the APAPANE camera.
    box: list
        [top, bottom, left, right]
    box_halfwidth : int
        Halfwidth of the spectral box to sum over.
    dark : np.ndarray
        Dark frame.
    nframes : int, optional
        Number of frames to average over. The default is 100.
        
    Returns
    -------
    data : np.ndarray
        Data from the APAPANE camera.
    """

    top, bottom, left, right = box
    bright = apapane.multi_recv_data(nframes) 
    bright = np.array(bright, dtype=float)  # Convert to float to avoid overflow

    # Average over the 100 frames
    avg = np.mean(bright, axis = 0)
    
    avg = avg[3:-3, 3:-3]  # This crop is to remove the magic pixel

    if dark is not None: # i.e. no dark
        data = avg - dark  # Subtract the dark frame
    else:
        data = avg

    data = data[top:bottom, left:right]  # Crop to the spectral box
    return data

def doTTscan(mems, segment, segpositions, piston, apapane, dark, box, writer) -> None:
    # Move mems to start position

    tips, tilts = segpositions
    scan = np.zeros((len(tips), len(tilts)))

    size = len(tips)*len(tilts)

    pbar = tqdm.tqdm(desc="Tip-tilt scan", total=size)

    # loop through tips
    for i, tip in enumerate(tips):

        # loop through tilts
        for j, tilt in enumerate(tilts):
            
            pbar.write(f'{tip=} {tilt=}') # figure out how to format this

            # Prepare DM command
            tip = float(tip)
            tilt = float(tilt)
            
            # Get time
            # dt = str(datetime.now())
            t = time.time()

            # Set the position
            mems.set_segment(segment, piston, tip, tilt)

            # Get the position
            pttpos = mems.get_ptt()

        
            # Write timestamp and positions to file
            writer.writerow(["Timestamp:", t])
            writer.writerow(["ptt:", pttpos])
            saveframe(apapane, segment, tip, tilt, t)

            data = getdata(apapane, box, dark)

            # collect data sum
            scan[i,j] = np.sum(data)

            
            
            pbar.update()

            time.sleep(0.5)
    
    mems.set_segment(segment, piston, 0.0, 0.0)

    
    return scan









    
