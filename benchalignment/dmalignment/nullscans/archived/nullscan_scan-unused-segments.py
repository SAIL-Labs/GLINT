import sys

# import apiMEMsControl
import shmDMcontrol
from pyMilk.interfacing.shm import SHM
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import time
import tqdm
import json
import os
from datetime import datetime, timezone
import glint_paths



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
        dark = hdul[0].data#[0, 3:-3, 3:-3]  # This crop is to remove the magic pixel
        dark = np.array(np.mean(dark, axis=0), dtype=float)  # Convert to float to avoid overflow

    return dark

def open_devices(): 
    apapane = SHM('apapane')
    dm = shmDMcontrol.DM()

    return apapane, dm

        
def save_data(segment: list, movie, opd, timestamps, savepath, avg:bool) -> None:


    # Primary HDU (empty header or optional metadata)
    hdu0 = fits.PrimaryHDU()

    # Image HDU for the movie (multi-frame)
    hdu1 = fits.ImageHDU(movie, name='MOVIE')

    # Table HDU for OPD and Timestamps
    col_opd = fits.Column(name='OPD', array=opd, format='E')          # 32-bit float
    col_time = fits.Column(name='TIMESTAMP', array=timestamps, format='D')  # 64-bit float
    hdu2 = fits.BinTableHDU.from_columns([col_opd, col_time], name='METADATA')

    # Write to file
    hdul = fits.HDUList([hdu0, hdu1, hdu2])

    if avg:
        hdul.writeto(f'{savepath}/avgmovie_{segment}.fits', overwrite=True) # the timestamps will be from the last scan
    else:
        hdul.writeto(f'{savepath}/movie_{segment}.fits', overwrite=True)


def getdata(apapane, box, dark, nframes = 50) -> np.ndarray:
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
    
    
    data = avg - dark  # Subtract the dark frame
    data = data[top:bottom, left:right]  # Crop to the spectral box

    return data

def nullscan(dm, segment,  pistpositions, box, dark) -> None:

    n = len(pistpositions)
    nullscan_seg = -1*np.zeros(n)

    # Set the two segments to their maximum piston values to start.
    dm.set_segment(segment, pistpositions[0], 0, 0) 
    time.sleep(0.01)

    print(f'scanning {segment}')

    height = box[1] - box[0] # for cred2
    width = box[3] - box[2]
    movie_len = n 
    movie = np.zeros((movie_len, height, width))
    timestamps = []

    pbar = tqdm.tqdm(desc="Null scan", total=2*n - 1)

    
    # 1. Loop over 2*n values.
    # 2. For the first n, keep segment 2 at its minimum piston value and scan segment 1 form maximum to minimum.
    # 3. Then at iteration n, segment 1 should be at a minimum position so move on to scanning over segment 2 from minimum to maximum
    for posnum in range(movie_len):
        frame = getdata(apapane, box, dark)
        movie[posnum] = frame

        newpos = pistpositions[posnum] # New segment position
        dm.set_segment(segment, newpos, 0, 0)
        time.sleep(0.001)

        # Get data, sum over the spectral box, and store the summed flux
        nullscan_seg[posnum] = np.sum(frame)        
        
        pbar.update()


    # Store the summed data and the segment positions for each scan
    scan = nullscan_seg
    opd = pistpositions - pistpositions[-1]

    # save movie as fits file
    # save_data(segment, movie, opd, timestamps, savepath, avg = False)

    dm.set_segment(segment, 0.0, 0.0, 0.0)

    return movie, scan, opd, timestamps


def scan(dm, dmpositions: np.ndarray, apapane: SHM, dark: np.ndarray, boxes: list, segment: int) -> None:
    # boxes: list of [top,bottom,left,right] for the 3 photometry channels

    boxheight = boxes[0][1] - boxes[0][0]
    boxwidth = boxes[0][3] - boxes[0][2]
    nchan = len(boxes)

    nsteps = [len(dmpositions[0]), len(dmpositions[1])]
    totalsteps = nsteps[0]*nsteps[1]

    # 4D cube: [step, channel, y, x]
    scan = np.zeros((totalsteps, nchan, boxheight, boxwidth)) 
    timestamps = []

    pbar = tqdm.tqdm(desc=f"Tip-tilt scan seg {segment}", total=totalsteps)

    total = 0
    for tip in dmpositions[0]:
        for tilt in dmpositions[1]:
            dm.set_segment(segment, 0.0, float(tip), float(tilt))

            # grab all three photometric channels
            for i, box in enumerate(boxes):
                data = getdata(apapane, box, dark)
                scan[total, i] = data

            timestamps.append(time.time())
            time.sleep(0.1)

            pbar.update()
            total += 1

    # reset segment
    dm.set_segment(segment, 0.0, 0.0, 0.0)

    return scan, timestamps



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


def plot_scan(segment, scan, opd, iteration:int, savepath, save:bool, avg:bool) -> None:

    normscan = scan/np.max(scan)
    # plt.figure()
    plt.plot(opd, normscan)
    plt.xlabel('OPD (um)')
    plt.ylabel('Normalised summed flux')
    plt.title(f'Null scan for segment {segment}')
    plt.grid(True)
    if save:
        if avg:
            plt.savefig(f'{savepath}/avg_segment_{segment}_scan{iteration}.png')
            # plt.show()
        else:
            plt.savefig(f'{savepath}/segment_{segment}_scan{iteration}.png')
    
    plt.clf()
    

if __name__ == "__main__":

    # Load settings
    with open('scanparameters.json', 'r') as f:
        params = json.load(f)

    # Now you can access:
    date = params['date']
    iteration = params['iteration']
    numavg = params['numavg']
    step_size = np.array(params['step_size'])
    scan_range = np.array(params['scan_range'])
    start_pos = np.array(params['start_pos'])
    nullpeaks = params['nullpeaks']
    boundingvals = params['boundingvals']
    box_halfwidth = params['box_halfwidth']
    iscred1 = params['iscred1']

   



   # Get the dark frame
    dark_filepath = str(glint_paths.DATA_ROOT / 'alignment_scans' / 'darknull.fits')
    dark = getdark(dark_filepath)

    savepath = str(glint_paths.data_dir('alignment_scans', 'nullscans', '2025', date, f'scan{iteration}'))
    checksavepath(savepath)

    nsteps = np.ceil(scan_range/step_size).astype(int) + 1  # Number of steps in the scan (need to plus 1 to include the last position)
    
    
    pistpositions = np.linspace(start_pos, start_pos + scan_range, nsteps)

    # Open the APAPANE camera and DM
    apapane, dm = open_devices()


    for segment in range(37):

        avgscan = []
        avgmovie = []

        for num in range(numavg):

            # Get the spectral box for the null scan, [top, bottom, left, right]
            box = getbox(segment, nullpeaks, boundingvals, box_halfwidth, iscred1)

            # Perform the null scan
            movie, scan, opd, timestamps = nullscan(dm, segment, pistpositions, box, dark)
        
            # Plot and save the data
            plot_scan(segment, scan, opd, iteration, savepath,save = True, avg = False)

            avgscan.append(scan)
            avgmovie.append(movie)
        
        avgscan = np.mean(avgscan, axis = 0)
        avgmovie = np.array(avgmovie)
        print(avgmovie.shape)
        avgmovie = np.mean(avgmovie, axis = 0)
        print(avgmovie.shape)

        plot_scan(segment, avgscan, opd, iteration, savepath, save = True, avg = True)

        save_data(segment, avgmovie, opd, timestamps, savepath, avg = True)

# if __name__ == '__main__':
#     with open('scanparameters.json', 'r') as f:
#         params = json.load(f)

#     date = params['date']
#     iteration = params['iteration']
#     step_size = np.array(params['step_size'])
#     scan_range = np.array(params['scan_range'])
#     start_pos = np.array(params['start_pos'])
#     peaks = params['peaks']
#     boundingvals = params['boundingvals']
#     box_halfwidth = params['box_halfwidth']
#     iscred1 = params['iscred1']

#     dark_filepath = str(glint_paths.DATA_ROOT / 'alignment_scans' / 'dark.fits')
#     dark = getdark(dark_filepath)

#     savepath = str(glint_paths.data_dir('alignment_scans', 'nullscans', '2025', date, f'scan{iteration}'))
#     checksavepath(savepath)

#     # get the three photometry boxes
#     boxes = [getbox(peak, boundingvals, box_halfwidth, iscred1) for peak in peaks]  

#     nsteps = np.ceil(scan_range/step_size).astype(int) + 1
#     dmpositions = [np.linspace(start_pos[i], start_pos[i] + scan_range[i], nsteps[i]) for i in range(2)]
#     dmpositions = np.array(dmpositions)

#     apapane, dm = open_devices()  

#     for segment in range(37):
 
#         try:
#             scanned_cube, timestamps = scan(dm, dmpositions, apapane, dark, boxes, segment)

#             primary_hdu = fits.PrimaryHDU()
#             phot_hdu = fits.ImageHDU(data=scanned_cube, name='PHOTOMETRY')
#             mount_hdu = fits.ImageHDU(data=dmpositions, name='DM_POS')
#             timestamps_hdu = fits.ImageHDU(data=np.array(timestamps), name='TIMESTAMPS')

#             hdul = fits.HDUList([primary_hdu, phot_hdu, mount_hdu, timestamps_hdu])
#             filename = f'{savepath}/nullscan_seg{segment}_{iteration}.fits'
#             hdul.writeto(filename, overwrite=True)
#             print(f'Saved {filename}')

#         except Exception as e:
#             import traceback
#             traceback.print_exc()
#             sys.exit(1)




