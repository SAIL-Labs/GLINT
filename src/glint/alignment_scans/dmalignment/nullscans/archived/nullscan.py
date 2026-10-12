import sys

import shmDMcontrol
from pyMilk.interfacing.shm import SHM
import numpy as np
from astropy.io import fits
import time
import matplotlib.pyplot as plt
import os
import json
import glint_paths

# Tip and tilt values fo segments [11,20,31] in that order


def getdata(apapane, box, dark, nframes = 100) -> np.ndarray:
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
    
    # avg = avg[3:-3, 3:-3]  # This crop is to remove the magic pixel
    data = avg - dark  # Subtract the dark frame
    data = data[top:bottom, left:right]  # Crop to the spectral box

    # plt.imshow(data)
    # plt.show()
    
    return data



def nullscan(apapane, dm, segment: int, tip: float, tilt: float, pistpositions: np.ndarray, box, dark):
    '''
    get the segment to scan over
    start loop
    set segment pos
    get data
    add sum to an array
    return array
    '''

    nullscan = np.array([])

    # pbar = tqdm.tqdm(desc="Null scan", total=len(pistpositions))

    for pist in pistpositions:

        dm.set_segment(segment, pist, tip, tilt)
        time.sleep(0.001)

        data = getdata(apapane, box, dark)
        sumflux = np.sum(data)
        nullscan = np.append(nullscan, sumflux)
        # return nullscan
    
    dm.set_segment(segment, 0, tip, tilt)

    
    return nullscan



def plot_scan(baseline, pistpositions, scan, iteration:int, savepath) -> None:
    '''
    Get the pistpositions
    Get the scanvec
    plot position vs scanvec

    '''
    plt.figure()
    opd = (pistpositions - pistpositions[0])*2
    normscan = np.abs(scan)/np.max(np.abs(scan))
    plt.plot(opd, normscan)
    plt.xlabel('OPD (um)')
    plt.ylabel('Normalised summed flux')
    plt.title(f'Null scan of segment {baseline[0]} for baseline {baseline}')

    plt.savefig(f'{savepath}/segment1_{baseline[0]}:{baseline[1]}_scan{iteration}.png')
    plt.clf()




def tiltunused(dm):
    segments = [11,20,31]

    for seg in range(37):
        if seg not in segments:
            dm.set_segment(seg, 0, 4, -2)


def getbox(baseline, nullpeaks, boundingvals,  box_halfwidth, iscred1):
    """
    Get the spectral box for the null scan.

    Parameters
    ----------
    baseline : list
        The two segments to scan over.
    nullpeaks : list
        Centre of the spectral boxes.
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

    null1, null2, null3 = nullpeaks
    val1, val2 = boundingvals

    # Assign the null channel spectral peak value based on the baseline 
    if set(baseline) == {"11", "20"}:
        null = null2
    elif set(baseline) == {"11", "31"}:
        null = null1
    elif set(baseline) == {"20", "31"}:
        null = null3

    # If True, the spectra are horizontal, if False, the spectra are vertical
    if iscred1:
        top = null-box_halfwidth
        bottom = null+box_halfwidth
        left = val1
        right = val2
    else:
        left = null-box_halfwidth
        right = null+box_halfwidth
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
        dark = hdul[0].data#[0, 3:-3, 3:-3]  # This crop is to remove the magic pixel
        dark = np.array(np.mean(dark), dtype=float)  # Convert to float to avoid overflow

    return dark


def open_devices():
    """
    Open the APAPANE camera and dm.

    Returns
    -------
    apapane : SHM
        SHM object for the APAPANE camera.
    dm : shmDMcontrol.DM()
        DM object.
    """

    apapane = SHM('apapane')
    dm = shmDMcontrol.DM()

    return apapane, dm



def checksavepath(savepath):
    # i want to make sure it doesnt already exist, and if it does, increase the iteration number saved in the json file. if it doesn exist, make a new directory for it
    import os
    if os.path.isdir(savepath):
        # check if it is empty

        if os.listdir(savepath):
            print(f"Directory {savepath} already exists and is not empty!")
            # increase iteration number in json file
            with open('scanparameters.json', 'r') as f:
                params = json.load(f)
            iteration = params['iteration']
            iteration += 1
            params['iteration'] = iteration
            with open('scanparameters.json', 'w') as f:
                json.dump(params, f)
            print(f"Iteration number increased to {iteration}. Edit scanparameters.json directly if incorrect.")
            sys.exit()
            
        else:
            print("Directory already exists but is empty. Using this directory.")

    else:
        os.makedirs(savepath)
        print(f"Directory {savepath} created.\n")



if __name__ == "__main__":

    # Load settings
    with open('scanparameters.json', 'r') as f:
        params = json.load(f)

    # Now you can access:
    date = params['date']
    iteration = params['iteration']
    null = params['null']
    step_size = np.array(params['step_size'])
    scan_range = np.array(params['scan_range'])
    start_pos = np.array(params['start_pos'])
    nullpeaks = params['nullpeaks']
    tips = params['tips']
    tilts = params['tilts']
    boundingvals = params['boundingvals']
    box_halfwidth = params['box_halfwidth']
    iscred1 = params['iscred1']


   # Get the dark frame
    dark_filepath = str(glint_paths.DATA_ROOT / 'alignment_scans' / 'dark.fits')
    dark = getdark(dark_filepath)

    savepath = str(glint_paths.data_dir('alignment_scans', 'nullscans', '2025', date, f'scan{iteration}'))
    checksavepath(savepath)

    # Get the positions
    nsteps = np.ceil(scan_range/step_size).astype(int) + 1  # Number of steps in the scan (need to plus 1 to include the last position)
    pistpositions = np.linspace(start_pos, start_pos + scan_range, nsteps)

    # Open the APAPANE camera and DM
    apapane, dm = open_devices()

    try:
        tiltunused(dm)

        for baseline in [["11","31"], ["11","20"], ["20","31"]]:

            # Get the tip and tilt values for the segments
            seg1, _ = baseline
            segment = int(seg1)

            tip = tips[seg1]
            tilt = tilts[seg1]
            

            # Get the spectral box for the null scan, [top, bottom, left, right]
            box = getbox(baseline, nullpeaks, boundingvals, box_halfwidth, iscred1)

            # Perform the null scan
            scan = nullscan(apapane, dm, segment, tip, tilt, pistpositions, box, dark)

            # Plot and save the data
            plot_scan(baseline, pistpositions, scan, iteration, savepath)
            # save_data(baseline, scan, iteration)

    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        sys.exit(1)


