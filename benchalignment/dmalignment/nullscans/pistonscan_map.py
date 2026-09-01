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


# -------------------------
# Helpers
# -------------------------

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
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid box dimensions: top={top}, bottom={bottom}, left={left}, right={right}")

    bright = apapane.multi_recv_data(nframes) 
    bright = np.array(bright, dtype=float)  # Convert to float to avoid overflow
    avg = np.mean(bright, axis = 0) # Average over nframes
    data = avg - dark  # Subtract the dark frame
    return data[top:bottom, left:right]


def tilt_all_except(dm, keep: list):
    """
    Tilt all segments away except those in the `keep` list.

    Parameters
    ----------  
    dm : shmDMcontrol.DM
        DM object.
    keep : list of str or int
        Segments to leave untouched.
    """
    keep = set(str(k) for k in keep)
    for seg in range(37):
        if str(seg) not in keep:
            dm.set_segment(seg, 0, 3, 0)

def zero_dm(dm):
    
    """
    Zeros all segments 

    Parameters
    ----------  
    dm : shmDMcontrol.DM
        DM object.
    """
    for seg in range(37):
        dm.set_segment(seg, 0, 0, 0)

def set_segment_state(dm, segment, piston, tilt_unused, optimise_injection, tips, tilts):
    """
    Set one segment to its correct piston/tip/tilt state.

    Rules
    -----
    - If optimise_injection is True and segment is one of [11, 20, 31],
      use the calibrated tip/tilt values from tips/tilts.
    - Otherwise, if tilt_unused is True, steer the segment off with (3, 0).
    - Otherwise leave it flat with (0, 0).
    """
    if optimise_injection and segment in [11, 20, 31]:
        dm.set_segment(segment, piston, tips[str(segment)], tilts[str(segment)])
    elif tilt_unused:
        dm.set_segment(segment, piston, 3, 0)
    else:
        dm.set_segment(segment, piston, 0, 0)

def initialise_dm(dm, tilt_unused, optimise_injection, tips, tilts):
    """
    Initialise all 37 segments into their parked state.
    """
    for seg in range(37):
        set_segment_state(
            dm=dm,
            segment=seg,
            piston=0,
            tilt_unused=tilt_unused,
            optimise_injection=optimise_injection,
            tips=tips,
            tilts=tilts,
        )

def plot_scan(segment, scans, opd, minimums, maximums, iteration:int, savepath, save:bool, normalise = True,) -> None:
   
    
    for i in np.arange(3):

        if normalise:
            scan = scans[i]/np.max(scans[i])
        else:
            scan = scans[i]
            plt.ylim([minimums[i], maximums[i]])

        plt.plot(opd, scan)
        plt.xlabel('OPD (um)')
        
        
        plt.title(f'Piston scan for segment {segment} for photometry {i+1}')
        plt.grid(True)

        if normalise:
            plt.ylabel('Normalised summed flux')
            plt.savefig(f'{savepath}/normsegment{segment}_phot{i+1}_scan{iteration}.png')
        
        else:
            plt.ylabel('Summed flux')
            plt.savefig(f'{savepath}/segment{segment}_phot{i+1}_scan{iteration}.png')
        
        
        plt.clf()


    



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
        null = null1; val1, val2 = boundingvals[0]
    elif set(baseline) == {"11", "20"}:
        null = null2; val1, val2 = boundingvals[1]
    elif set(baseline) == {"20", "31"}:
        null = null3; val1, val2 = boundingvals[2]
    else:
        raise ValueError(f"Unknown baseline {baseline} for nullpeaks/boundingvals mapping.")

    # If True, the spectra are horizontal, if False, the spectra are vertical
    if iscred1:
        top, bottom = null - box_halfwidth, null + box_halfwidth
        left, right = val1, val2
    else:
        left, right = null - box_halfwidth, null + box_halfwidth
        top, bottom = val1, val2

    box = [int(top), int(bottom), int(left), int(right)]

    return box

def getboxes(nullpeaks, boundingvals,  box_halfwidth):
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

    top1, bottom1 = null1 - box_halfwidth, null1 + box_halfwidth
    left1, right1 = boundingvals[0]

    top2, bottom2 = null2 - box_halfwidth, null2 + box_halfwidth
    left2, right2 = boundingvals[0]

    top3, bottom3 = null3 - box_halfwidth, null3 + box_halfwidth
    left3, right3 = boundingvals[0]

    box1 = [int(top1), int(bottom1), int(left1), int(right1)]
    box2 = [int(top2), int(bottom2), int(left2), int(right2)]
    box3 = [int(top3), int(bottom3), int(left3), int(right3)]

    return [box1, box2, box3]
    
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
        os.makedirs(savepath)
        print(f"Directory {savepath} created.\n")

# -------------------------
# Core scan
# -------------------------

def nullscan(apapane, dm, segment, pistpositions,
             boxes, dark, savepath, nframes=1,
             tilt_unused=True, optimise_injection=False,
             tips=None, tilts=None) -> None:

    box1, box2, box3 = boxes

    pistpositions = np.asarray(pistpositions, dtype=float)
    n = len(pistpositions)

    nullscan_phot1 = np.zeros(n, dtype=float)
    nullscan_phot2 = np.zeros(n, dtype=float)
    nullscan_phot3 = np.zeros(n, dtype=float)

    # Set initial position
    set_segment_state(
        dm=dm,
        segment=segment,
        piston=pistpositions[0],
        tilt_unused=tilt_unused,
        optimise_injection=optimise_injection,
        tips=tips,
        tilts=tilts,
    )

    time.sleep(0.01)

    nsteps = len(pistpositions)
    print(f'Scanning segment {segment}')

    pbar = tqdm.tqdm(desc="Null scan", total=nsteps)

    for posnum in range(nsteps):
        frame_phot1 = getdata(apapane, box1, dark, nframes)
        frame_phot2 = getdata(apapane, box2, dark, nframes)
        frame_phot3 = getdata(apapane, box3, dark, nframes)

        newpos = pistpositions[posnum]

        set_segment_state(
            dm=dm,
            segment=segment,
            piston=newpos,
            tilt_unused=tilt_unused,
            optimise_injection=optimise_injection,
            tips=tips,
            tilts=tilts,
        )

        time.sleep(0.001)

        nullscan_phot1[posnum] = np.sum(frame_phot1)
        nullscan_phot2[posnum] = np.sum(frame_phot2)
        nullscan_phot3[posnum] = np.sum(frame_phot3)

        pbar.update()

    scans = [nullscan_phot1, nullscan_phot2, nullscan_phot3]

    # Return scanned segment to parked state at piston = 0
    set_segment_state(
        dm=dm,
        segment=segment,
        piston=0,
        tilt_unused=tilt_unused,
        optimise_injection=optimise_injection,
        tips=tips,
        tilts=tilts,
    )

    opd = pistpositions - pistpositions[-1]
    opd = 2.0 * opd

    return scans, opd


def save_nullscan_table(savefile, all_rows, header_info=None, comments=None):
    all_rows = np.array(
        all_rows,
        dtype=[
            ('SEGMENT', 'i4'),
            ('PHOT_CHAN', 'i4'),
            ('OPD', 'f8'),
            ('FLUX', 'f8'),
            ('AVG_INDEX', 'i4'),
        ]
    )

    cols = fits.ColDefs([
        fits.Column(name='SEGMENT', format='J', array=all_rows['SEGMENT']),
        fits.Column(name='PHOT_CHAN', format='J', array=all_rows['PHOT_CHAN']),
        fits.Column(name='OPD', format='D', array=all_rows['OPD']),
        fits.Column(name='FLUX', format='D', array=all_rows['FLUX']),
        fits.Column(name='AVG_INDEX', format='J', array=all_rows['AVG_INDEX']),
    ])

    primary_hdu = fits.PrimaryHDU()
    if header_info is not None:
        for key, value in header_info.items():
            if key.upper() not in ['SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND']:
                primary_hdu.header[key[:8].upper()] = value
    
    if comments is not None:
        for c in comments:
            primary_hdu.header.add_comment(c)

    table_hdu = fits.BinTableHDU.from_columns(cols, name='SCAN_DATA')
    fits.HDUList([primary_hdu, table_hdu]).writeto(savefile, overwrite=True)

# -------------------------
# Main
# -------------------------

if __name__ == "__main__":

    tilt_unused=True
    optimise_injection = True

    comments = [
        "Injection optimised on segments 11,20,31",
        "Unused segments set to (p,t,t)=(0,0,0)",
        "z mount position 1000"
    ]

    
    # Open the APAPANE camera and DM
    apapane, dm = open_devices()

    # Load settings
    with open('scanparameters.json', 'r') as f:
        params = json.load(f)

    # Now you can access:
    date = params['date']
    year = params['year']
    iteration = params['iteration']
    nframes = params['nframes']
    numavg = params['numavg']
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

    savepath = str(glint_paths.data_dir('alignment_scans', 'nullscans', year, date, f'scan{iteration}'))
    checksavepath(savepath)


    nsteps = np.ceil(scan_range/step_size).astype(int) + 1  # Number of steps in the scan (need to plus 1 to include the last position)
    pistpositions = np.linspace(start_pos, start_pos + scan_range, nsteps)

    initialise_dm(
        dm=dm,
        tilt_unused=tilt_unused,
        optimise_injection=optimise_injection,
        tips=tips,
        tilts=tilts,
    )
            
    
    all_rows = []

    for avg_index in range(numavg):
        for segment in np.arange(37):
            boxes = getboxes(nullpeaks, boundingvals, box_halfwidth)

            scan, opd = nullscan(
                apapane=apapane, dm=dm, segment=segment,
                pistpositions=pistpositions, boxes=boxes,
                dark=dark, savepath=savepath, nframes=nframes,
                tilt_unused=tilt_unused,
                optimise_injection=optimise_injection,
                tips=tips, tilts=tilts,
            )

            for phot_chan in range(3):
                for i in range(len(opd)):
                    all_rows.append((
                        int(segment),
                        int(phot_chan + 1),
                        float(opd[i]),
                        float(scan[phot_chan][i]),
                        int(avg_index)
                    ))

    header_info = {
        'DATE': str(date),
        'YEAR': int(year),
        'ITER': int(iteration),
        'NFRAME': int(nframes),
        'NUMAVG': int(numavg),
    }

    savefile = os.path.join(savepath, f'nullscan_table_{iteration}.fits')
    save_nullscan_table(savefile, all_rows, header_info=header_info, comments=comments)

