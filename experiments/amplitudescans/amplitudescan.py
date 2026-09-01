import sys

from hardware_control.dmcontrol import shmDMcontrol
from pyMilk.interfacing.shm import SHM
import numpy as np
from astropy.io import fits
import time
import matplotlib.pyplot as plt
import os
import json
import tqdm
import glint_paths

# Tip and tilt values fo segments [11,20,31] in that order


def getdata(apapane, box=None, dark=None, nframes=20, applybox = True) -> np.ndarray:
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

    bright = apapane.multi_recv_data(nframes)
    bright = np.array(bright, dtype=float)
    avg = np.mean(bright, axis=0)
    data = avg - dark

    if applybox:
        top, bottom, left, right = box
        data = data[top:bottom, left:right]

    return data




def nullscan(dm, baseline: np.ndarray, tip: np.ndarray, tilt: np.ndarray, pistpositions, box, dark) -> None:

    seg1, seg2 = int(baseline[0]), int(baseline[1])
    tip_seg1, tip_seg2 = tip
    tilt_seg1, tilt_seg2 = tilt
    n = len(pistpositions)
    print(f'tip_seg1 = {tip_seg1}, tilt_seg1 = {tilt_seg1}')
    print(f'tip_seg2 = {tip_seg2}, tilt_seg2 = {tilt_seg2}')

    nullscan_seg1 = -1*np.zeros(n)
    nullscan_seg2 = -1*np.zeros(n - 1)

    # Positions of the segments for each scan. The segment not being scanned will have constant values.
    # For the first scan, segment 2 wlil be at a minimum and sgement 1 will piston from maximum to minimum
    # For the second scan, segment 1 will be at a minimum and segment 2 will piston from minimum to maximum
    # Scan1 positions-----------------------------------------
    pos_seg1_scan1 = pistpositions # Min --> Max
    pos_seg2_scan1 = pistpositions[-1]

    # Scan2 positions-----------------------------------------
    pos_seg1_scan2 = pistpositions[-1]
    pos_seg2_scan2 = np.flip(pistpositions)[1:] # Max --> Min

    # pos_seg1_scan1 = pistpositions # Min --> Max
    # pos_seg2_scan1 = pistpositions[0]

    # # Scan2 positions-----------------------------------------
    # pos_seg1_scan2 = pistpositions[-1]
    # pos_seg2_scan2 = pistpositions # Min --> Max

    # Set the two segments to their maximum piston values to start.
    dm.set_segment(seg1, pos_seg1_scan1[0], tip_seg1, tilt_seg1) 
    time.sleep(0.01)
    dm.set_segment(seg2, pos_seg2_scan1, tip_seg2, tilt_seg2)
    time.sleep(0.01)

    print(f'scanning {seg1} and {seg2}')


    height = box[1] - box[0] # for cred2
    width = box[3] - box[2]

    pbar = tqdm.tqdm(desc="Null scan", total=2*n - 1)

    

    movie = np.zeros((2*n - 1, height, width))  # Store the data for each scan
    # 1. Loop over 2*n values.
    # 2. For the first n, keep segment 2 at its minimum piston value and scan segment 1 form maximum to minimum.
    # 3. Then at iteration n, segment 1 should be at a minimum position so move on to scanning over segment 2 from minimum to maximum
    for posnum in range (2*n - 1):

        # Scan 1 -----------------------------------------
        if posnum < n:
            frame = getdata(apapane, box, dark)
            movie[posnum] = frame

            # New segment position
            newpos_seg1 = pos_seg1_scan1[posnum]
            dm.set_segment(seg1, newpos_seg1, tip_seg1, tilt_seg1)
            time.sleep(0.001)

            # Get data, sum over the spectral box, and store the summed flux
            summed_flux = np.sum(frame)
            nullscan_seg1[posnum] = summed_flux

        # Scan 2 -----------------------------------------
        else:
            frame = getdata(apapane, box, dark)
            movie[posnum] = frame
            posnum = posnum - n  # Reset posnum to start from 0 again when scanning over segment 2
            
            # New segment position
            newpos_seg2 = pos_seg2_scan2[posnum]
            dm.set_segment(seg2, newpos_seg2, tip_seg2, tilt_seg2)
            time.sleep(0.001)

            # Get data, sum over the spectral box, and store the summed flux
            summed_flux = np.sum(frame)
            nullscan_seg2[posnum] = summed_flux
        
        pbar.update()
        
        # break


    # Store the summed data and the segment positions for each scan
    scan = np.concatenate([nullscan_seg1, nullscan_seg2])

    opd1 = pistpositions - pistpositions[-1]
    opd2 = np.abs(np.flip(pistpositions)[1:] - np.flip(pistpositions)[0])
    opd = np.concatenate([opd1, opd2])*2

    
    

    return scan, opd, movie


    
def plot_scan(baseline:list, scan, opd, iteration:int, savepath, save:bool) -> None:
    seg1, seg2 = baseline
    # normscan = scan/np.max(scan)
    plt.figure()
    plt.plot(opd, scan)
    plt.xlabel('OPD (μm)')
    plt.ylabel('Summed flux')
    plt.title(f'Null scan for baseline {seg1}-{seg2}')
    plt.grid(True)
    if save:
        plt.savefig(os.path.join(savepath, f'segment1_{seg1}_{seg2}_scan{iteration}.png'))
    plt.close()




def tiltunused(dm):
    segments = [11,20,31]

    for seg in range(37):
        if seg not in segments:
            dm.set_segment(seg, 0, 0, 6)

def save_data(baseline:list, scan, opd, savepath) -> None:
    seg1, seg2 = baseline

    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(scan, name='SCAN')
    col = fits.Column(name='OPD', array=opd, format='E')
    hdu2 = fits.BinTableHDU.from_columns([col], name='METADATA')

    hdul = fits.HDUList([hdu0, hdu1, hdu2])
    fits_path = os.path.join(savepath, f'scan{seg1}_{seg2}.fits')
    hdul.writeto(fits_path, overwrite=True)


def save_movie(baseline:list, movie, opd, savepath) -> None:
    seg1, seg2 = baseline

    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(movie, name='MOVIE')
    col = fits.Column(name='OPD', array=opd, format='E')
    hdu2 = fits.BinTableHDU.from_columns([col], name='METADATA')

    hdul = fits.HDUList([hdu0, hdu1, hdu2])
    fits_path = os.path.join(savepath, f'movie{seg1}_{seg2}.fits')
    hdul.writeto(fits_path, overwrite=True)

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

    # Assign the null channel spectral peak value based on the baseline 
    if set(baseline) == {"11", "31"}:
        null = null1
        val1, val2 = boundingvals[0]
    elif set(baseline) == {"11", "20"}:
        null = null2
        val1, val2 = boundingvals[1]
    elif set(baseline) == {"20", "31"}:
        null = null3
        val1, val2 = boundingvals[2]

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
        dark = hdul[0].data #[0, 3:-3, 3:-3]  # This crop is to remove the magic pixel
        dark = np.array(np.mean(dark, axis = 0), dtype=float)  # Convert to float to avoid overflow

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
        
        print(f"Directory {savepath} created.\n")


def scanamplitudes(box, tip, tilt, baseline, pistpositions, dark, apapane, dm, iteration, savepath):
    tilts_to_scan_seg1 = np.arange(tilt[0] - 1.3, tilt[0] + 0.2, 0.1)
    tilts_to_scan_seg2 = np.arange(tilt[1]+1, tilt[1] + 2.5, 0.1)

    for t1, tilt_seg1 in enumerate(tilts_to_scan_seg1):
        for t2, tilt_seg2 in enumerate(tilts_to_scan_seg2):

            tilts_rounded = [round(tilt_seg1, 3), round(tilt_seg2, 3)]
            scan, opd, movie = nullscan(dm, baseline, tip, tilts_rounded, pistpositions, box, dark)



            # scan, opd = nullscan(dm, baseline, tip, [tilt_seg1, tilt_seg2], pistpositions, box, dark)

            # Create subdirectory for each tilt combo
            savepath_tilt = os.path.join(savepath, f"tilt1_{tilt_seg1:.3f}_tilt2_{tilt_seg2:.3f}")
            os.makedirs(savepath_tilt, exist_ok=True)

            print(f'Saving data for baseline {baseline} with tilt values {tilt_seg1}, {tilt_seg2}')

            # Save the scan movie and OPD
            save_data(baseline, scan, opd, savepath_tilt)
            save_movie(baseline, movie, opd, savepath_tilt)

            # Save a representative apapane frame
            frame = getdata(apapane, dark = dark,applybox=False)
            fits.writeto(os.path.join(savepath_tilt, 'apapane_frame.fits'), frame, overwrite=True)

            # Save plot for this tilt combo
            plot_scan(baseline, scan, opd, iteration=iteration, savepath=savepath_tilt, save=True)

    

if __name__ == "__main__":

    # Load settings
    with open('scanparameters.json', 'r') as f:
        params = json.load(f)

    # Now you can access:
    date = params['date']
    iteration = params['iteration']
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
    dark_filepath = str(glint_paths.DATA_ROOT / 'alignment_scans' / 'darknull.fits')
    dark = getdark(dark_filepath)


    savepath = f'{glint_paths.data_dir("alignment_scans", "amplitudescans", "2025", date, f"scan{iteration}")}/'
    checksavepath(savepath)

    nsteps = np.ceil(scan_range/step_size).astype(int) + 1  # Number of steps in the scan (need to plus 1 to include the last position)
    pistpositions = np.linspace(start_pos, start_pos + scan_range, nsteps)

    # Open the APAPANE camera and DM
    apapane, dm = open_devices()

    tiltunused(dm)

    for baseline in [["11","31"]]:

        # Get the tip and tilt values for the segments
        seg1, seg2 = baseline
        tip = [tips[seg1], tips[seg2]]
        tilt = [tilts[seg1], tilts[seg2]]

        # Get the spectral box for the null scan, [top, bottom, left, right]
        box = getbox(baseline, nullpeaks, boundingvals, box_halfwidth, iscred1)

        scanamplitudes(box, tip, tilt, baseline, pistpositions, dark, apapane, dm, iteration, savepath)

   


        

        
 