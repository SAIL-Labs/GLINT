import sys
sys.path.append('/home/scexao/glint/control-code/')

import shmDMcontrol
from pyMilk.interfacing.shm import SHM
import numpy as np
from astropy.io import fits
import time
import matplotlib.pyplot as plt
import os
import json
import tqdm
from datetime import datetime, timezone


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
            dm.set_segment(seg, 0, -5.5, -4)

def save_data(baseline: list, movie, opd, timestamps, savepath, avg:bool) -> None:
    seg1, seg2 = baseline
    hdu0 = fits.PrimaryHDU() # Primary HDU (empty header or optional metadata)
    hdu1 = fits.ImageHDU(movie, name='MOVIE') # Image HDU for the movie (multi-frame)

    # Ensure timestamps length matches OPD length
    if timestamps is None or len(timestamps) != len(opd):
        timestamps = np.full(len(opd), np.nan, dtype=float)

    # Table HDU for OPD and Timestamps
    col_opd = fits.Column(name='OPD', array=opd, format='E')          # 32-bit float
    col_time = fits.Column(name='TIMESTAMP', array=timestamps, format='D')  # 64-bit float
    hdu2 = fits.BinTableHDU.from_columns([col_opd, col_time], name='METADATA')

    # Write to file
    hdul = fits.HDUList([hdu0, hdu1, hdu2])

    if avg:
        hdul.writeto(f'{savepath}/avgmovie_{seg1}:{seg2}.fits', overwrite=True) # the timestamps will be from the last scan
    else:
        hdul.writeto(f'{savepath}/movie_{seg1}:{seg2}.fits', overwrite=True)

def plot_scan(baseline:list, scan, opd, iteration:int, savepath, save:bool, avg:bool) -> None:

    plt.plot(opd, scan)
    plt.xlabel('OPD (um)')
    plt.ylabel('Summed flux')
    plt.title(f'Narrowband piston scan for aperture {baseline[0]}')
    plt.grid(True)
    if save:
        if avg:
            plt.savefig(f'{savepath}/avg_segments_{seg1}:{seg2}_scan{iteration}.png')
            # plt.show()
        else:
            plt.savefig(f'{savepath}/segments_{seg1}:{seg2}_scan{iteration}.png')
    
    plt.clf()
    
def plot_normscan(baseline:list, scan, opd, iteration:int, savepath, save:bool, avg:bool) -> None:

    normscan = scan/np.max(scan)

    plt.plot(opd, normscan)

    plt.xlabel('OPD (um)')
    plt.ylabel('Normalised summed flux')
    plt.title(f'Narrowband piston scan for aperture {baseline[0]}')
    plt.grid(True)
    if save:
        if avg:
            plt.savefig(f'{savepath}/normalised_avg_segments_{seg1}:{seg2}_scan{iteration}.png')
            # plt.show()
        else:
            plt.savefig(f'{savepath}/normalised_segments_{seg1}:{seg2}_scan{iteration}.png')
    
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


PARAM_FILE = '/home/scexao/glint/alignment_scans/nullscans/scanparameters.json'

def load_params():
    with open(PARAM_FILE, 'r') as f:
        return json.load(f)


def save_params(params):
    with open(PARAM_FILE, 'w') as f:
        json.dump(params, f, indent=2)


def checksavepath(base_dir):
    """
    Create/use a scan directory.

    If scan{iteration} already exists and is not empty, increment iteration
    in scanparameters.json until an empty/new directory is found.
    """

    params = load_params()
    iteration = params['iteration']

    while True:
        savepath = os.path.join(base_dir, f'scan{iteration}')

        if os.path.isdir(savepath):
            if os.listdir(savepath):
                print(f"{savepath} exists and is not empty. Incrementing iteration.")
                iteration += 1
                params['iteration'] = iteration
                save_params(params)
                continue

            else:
                print(f"{savepath} exists but is empty. Using it.")
                params['iteration'] = iteration
                save_params(params)
                return savepath

        else:
            os.makedirs(savepath)
            print(f"{savepath} created.")
            params['iteration'] = iteration
            save_params(params)
            return savepath

# -------------------------
# Core scan
# -------------------------

def nullscan(apapane, dm, baseline: list, tip: list, tilt: list, pistpositions,
             box, dark, savepath, nframes=1, scan_one_segment=True, tilt_unused=True,
             save_timestamps=False) -> None:
    
    seg1, seg2 = int(baseline[0]), int(baseline[1])
    tip_seg1, tip_seg2 = float(tip[0]), float(tip[1])
    tilt_seg1, tilt_seg2 = float(tilt[0]), float(tilt[1])
    pistpositions = np.asarray(pistpositions, dtype=float)
    n = len(pistpositions)

    if tilt_unused:
        if scan_one_segment:
            tilt_all_except(dm, keep=[seg1])
        else:
            tilt_all_except(dm, keep=[11, 20, 31])

    print(f'tip_seg1 = {tip_seg1}, tilt_seg1 = {tilt_seg1}')
    print(f'tip_seg2 = {tip_seg2}, tilt_seg2 = {tilt_seg2}')

    nullscan_seg1 = np.zeros(n, dtype=float)
    nullscan_seg2 = None if scan_one_segment else np.zeros(n - 1, dtype=float)

    # Positions
    pos_seg1_scan1 = pistpositions                      # min -> max
    pos_seg2_scan1 = pistpositions[-1]                  # hold at max
    pos_seg1_scan2 = pistpositions[-1]                  # (not used)
    pos_seg2_scan2 = np.flip(pistpositions)[1:]         # max -> min (skip duplicate)

    # Set initial positions
    dm.set_segment(seg1, pos_seg1_scan1[0], tip_seg1, tilt_seg1)
    time.sleep(0.01)
    if not scan_one_segment:
        dm.set_segment(seg2, pos_seg2_scan1, tip_seg2, tilt_seg2)
        time.sleep(0.01)

    print(f'Scanning baseline {seg1}–{seg2}')
    height = box[1] - box[0]
    width  = box[3] - box[2]
    movie_len = n if scan_one_segment else (2*n - 1)
    movie = np.zeros((movie_len, height, width), dtype=float)
    timestamps = [] if save_timestamps else None

    pbar = tqdm.tqdm(desc="Null scan", total=movie_len)

    for posnum in range(movie_len):
        frame = getdata(apapane, box, dark, nframes)

        movie[posnum] = frame

        if posnum < n:
            # Scan 1: move seg1
            newpos_seg1 = pos_seg1_scan1[posnum]
            dm.set_segment(seg1, newpos_seg1, tip_seg1, tilt_seg1)
            time.sleep(0.001)
            nullscan_seg1[posnum] = np.sum(frame)
        else:
            # Scan 2: move seg2
            if not scan_one_segment:
                idx = posnum - n
                newpos_seg2 = pos_seg2_scan2[idx]
                dm.set_segment(seg2, newpos_seg2, tip_seg2, tilt_seg2)
                time.sleep(0.001)
                nullscan_seg2[idx] = np.sum(frame)

        if save_timestamps:
            timestamps.append(time.time())

        pbar.update()

    scan = nullscan_seg1 if scan_one_segment else np.concatenate([nullscan_seg1, nullscan_seg2])

    # OPD stitching
    opd1 = pistpositions - pistpositions[-1]            # centered so last = 0
    opd2 = np.abs(np.flip(pistpositions)[1:] - np.flip(pistpositions)[0])
    opd  = opd1 if scan_one_segment else np.concatenate([opd1, opd2])
    
    opd = 2.0 * opd  # To account for reflection

    # Save cube + metadata now (scan plots saved separately)
    # Ensure timestamps length is consistent
    ts = np.asarray(timestamps) if save_timestamps else np.full(movie_len, np.nan, dtype=float)
    save_data(baseline, movie, opd, ts, savepath, avg=False)

    return movie, scan, opd, ts

# -------------------------
# Main
# -------------------------

if __name__ == "__main__":

    scan_one_segment = False
    tilt_unused = True
    save_timestamps = False

    b1 = True  # ["11", "31"]
    b2 = True  # ["20", "11"]
    b3 = True  # ["31", "20"]

    # Load params
    params = load_params()

    # Get current UTC year and date
    now = datetime.now(timezone.utc)
    year = now.year
    date = now.strftime("%m-%d")

    # Reset iteration automatically if the date changed
    if params.get('date') != date or params.get('year') != year:
        params['iteration'] = 1

    params['date'] = date
    params['year'] = year
    save_params(params)

    # Build base directory up to the date folder
    base_dir = f'/home/scexao/glint/alignment_scans/nullscans/{year}/{date}'

    # This may update iteration in the JSON
    savepath = checksavepath(base_dir)

    # Reload params AFTER checksavepath has possibly updated them
    params = load_params()
    iteration = params['iteration']

    # Now use refreshed params everywhere below
    date = params['date']
    year = params['year']
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

    # Open the APAPANE camera and DM
    apapane, dm = open_devices()

    # Get the dark frame
    dark_filepath = '/home/scexao/glint/alignment_scans/darknull.fits'
    dark = getdark(dark_filepath)

    nsteps = np.ceil(scan_range / step_size).astype(int) + 1
    pistpositions = np.linspace(start_pos, start_pos + scan_range, nsteps)

    baseline_map = {
        'b1': ["11","31"],
        'b2': ["11","20"],
        'b3': ["20","31"],
    }
    to_scan = []
    if b1: to_scan.append(baseline_map['b1'])
    if b2: to_scan.append(baseline_map['b2'])
    if b3: to_scan.append(baseline_map['b3'])
    if not to_scan:
        print("No baselines selected (all b1/b2/b3 False). Exiting.")
        sys.exit(0)

    for baseline in to_scan:
        avgscan_list  = []
        avgmovie_list = []
        last_opd      = None
        last_ts       = None

        seg1, seg2 = baseline
        tip  = [tips[seg1],  tips[seg2]]
        tilt = [tilts[seg1], tilts[seg2]]

        box = getbox(baseline, nullpeaks, boundingvals, box_halfwidth, iscred1)

        for _ in range(numavg):
            movie, scan, opd, ts = nullscan(
                apapane, dm, baseline, tip, tilt, pistpositions, box, dark, savepath,
                nframes=nframes, scan_one_segment=scan_one_segment,
                tilt_unused=tilt_unused, save_timestamps=save_timestamps)
            
            plot_scan(baseline, scan, opd, iteration, savepath, save=True, avg=False)
            avgscan_list.append(scan)
            avgmovie_list.append(movie)
            last_opd = opd
            last_ts  = ts

        avgscan  = np.mean(np.stack(avgscan_list, axis=0),  axis=0)
        avgmovie = np.mean(np.stack(avgmovie_list, axis=0), axis=0)

        plot_scan(baseline, avgscan, last_opd, iteration, savepath, save=True, avg=True)
        plot_normscan(baseline, avgscan, last_opd, iteration, savepath, save=True, avg=True)
        save_data(baseline, avgmovie, last_opd, last_ts, savepath, avg=True)




# def save_data(baseline:list, scan, opd, timestamps, iteration, savepath) -> None:
#     seg1, seg2 = baseline


#     # HDU 0: Primary (empty or summary)
#     primary_hdu = fits.PrimaryHDU()

#     # HDU 1: 3D photometry image cube
#     null_hdu = fits.ImageHDU(data=scan, name='NULL')

#     # HDU 2: mount positions (saved as an image or table)
#     mount_hdu = fits.ImageHDU(data=opd, name='DM_POS')

#     timestamps_hdu = fits.ImageHDU(data=timestamps, name='TIMESTAMPS')


#     # Write to file
#     hdul = fits.HDUList([primary_hdu, null_hdu, mount_hdu, timestamps_hdu])
#     hdul.writeto(f'{savepath}/scan_{seg1}:{seg2}_{iteration}.fits')
#     print('saved')



# def nullscan(dm, baseline: np.ndarray, tip: np.ndarray, tilt: np.ndarray, pistpositions, box, dark, nframes = 1, scan_one_segment=False, tilt_unused=True, save_timestamps=False) -> None:

#     seg1, seg2 = int(baseline[0]), int(baseline[1])
#     tip_seg1, tip_seg2 = tip
#     tilt_seg1, tilt_seg2 = tilt
#     n = len(pistpositions)

#     if tilt_unused:
#         if scan_one_segment:
#             tilt_all_except(dm, keep=[seg1])  # Only scanning seg1
#         else:
#             tilt_all_except(dm, keep=[11, 20, 31])  # Scanning both

#     print(f'tip_seg1 = {tip_seg1}, tilt_seg1 = {tilt_seg1}')
#     print(f'tip_seg2 = {tip_seg2}, tilt_seg2 = {tilt_seg2}')

#     nullscan_seg1 = -1*np.zeros(n)
#     nullscan_seg2 = -1*np.zeros(n - 1) if not scan_one_segment else None

#     # Positions of the segments for each scan. The segment not being scanned will have constant values.
#     # For the first scan, segment 2 will be at a maximum and segment 1 will piston from minimum to maximum
#     # For the second scan, segment 1 will be at a maximum and segment 2 will piston from maximum to minimum
#     # Scan1 positions-----------------------------------------
#     pos_seg1_scan1 = pistpositions # Min --> Max
#     pos_seg2_scan1 = pistpositions[-1]

#     # Scan2 positions-----------------------------------------
#     pos_seg1_scan2 = pistpositions[-1]  # not used but just for logic when reading code
#     pos_seg2_scan2 = np.flip(pistpositions)[1:] # Max --> Min but skip the first position to avoid repeated steps

#     # Set the segments to their first position
#     dm.set_segment(seg1, pos_seg1_scan1[0], tip_seg1, tilt_seg1) 
#     time.sleep(0.01)

#     if not scan_one_segment:
#         dm.set_segment(seg2, pos_seg2_scan1, tip_seg2, tilt_seg2)
#         time.sleep(0.01)

#     print(f'scanning {seg1} and {seg2}')

#     height = box[1] - box[0] 
#     width = box[3] - box[2]
#     movie_len = n if scan_one_segment else 2 * n - 1
#     movie = np.zeros((movie_len, height, width))
#     timestamps = []

#     pbar = tqdm.tqdm(desc="Null scan", total=2*n - 1)

    
#     # 1. Loop over 2*n values.
#     # 2. For the first n, keep segment 2 at its minimum piston value and scan segment 1 form maximum to minimum.
#     # 3. Then at iteration n, segment 1 should be at a minimum position so move on to scanning over segment 2 from minimum to maximum
#     for posnum in range (movie_len):
#         frame = getdata(apapane, box, dark, nframes)
#         # check this!
#         plt.imshow(frame)
#         plt.show()
#         movie[posnum] = frame

#         # Scan 1 -----------------------------------------
#         if posnum < n:
            
#             newpos_seg1 = pos_seg1_scan1[posnum] # New segment position
#             dm.set_segment(seg1, newpos_seg1, tip_seg1, tilt_seg1)
#             time.sleep(0.001)

#             # Get data, sum over the spectral box, and store the summed flux
#             nullscan_seg1[posnum] = np.sum(frame)

#         # Scan 2 -----------------------------------------
#         elif not scan_one_segment:
#             idx = posnum - n  # Reset posnum to start from 0 again when scanning over segment 2
#             newpos_seg2 = pos_seg2_scan2[idx] # New segment position
#             dm.set_segment(seg2, newpos_seg2, tip_seg2, tilt_seg2)
#             time.sleep(0.001)

#             # Get data, sum over the spectral box, and store the summed flux
#             nullscan_seg2[idx] = np.sum(frame)
            
#         if save_timestamps:
#             timestamps.append(time.time())
        
#         pbar.update()


#     # Store the summed data and the segment positions for each scan
#     scan = nullscan_seg1 if scan_one_segment else np.concatenate([nullscan_seg1, nullscan_seg2])

#     opd1 = pistpositions - pistpositions[-1]
#     opd2 = np.abs(np.flip(pistpositions)[1:] - np.flip(pistpositions)[0])
#     opd = opd1 if scan_one_segment else np.concatenate([opd1, opd2])*2

#     # save movie as fits file
#     save_data(baseline, movie, opd, timestamps, savepath, avg = False)

#     return movie, scan, opd, timestamps

