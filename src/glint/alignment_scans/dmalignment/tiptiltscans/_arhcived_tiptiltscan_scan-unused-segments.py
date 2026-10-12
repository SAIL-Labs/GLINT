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


def tiltseg(dm, segments = [11, 20, 31]):

    for seg in range(37):
        if seg in segments:
            dm.set_segment(seg, 0, 0, 6)
        

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
    
    avg = avg#[3:-3, 3:-3]  # This crop is to remove the magic pixel
    data = avg - dark  # Subtract the dark frame
    data = data[top:bottom, left:right]  # Crop to the spectral box

    

    return data



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

    # plt.imshow(data)
    # plt.show()
    return scan, timestamps

# def scan(dm, dmpositions: np.ndarray, apapane: SHM, dark: np.ndarray, box: list) -> None:
#     # Move dm to start position

#     top, bottom, left, right = boxes[0]

#     boxheight = bottom-top
#     boxwidth = right-left

#     nsteps = [len(dmpositions[0]), len(dmpositions[1])]  # Number of steps in the scan per axis
#     totalsteps = nsteps[0]*nsteps[1]
#     scan = np.zeros((totalsteps, boxheight, boxwidth)) 
#     timestamps = []
    

#     # scan = np.zeros((len(dmpositions[0]), len(dmpositions[1])))

#     pbar = tqdm.tqdm(desc="Tip-tilt scan", total=totalsteps)

#     # loop through tips
#     total = 0
#     for tip in dmpositions[0]:


#         # print(f'tip: {tip}') # figure out how to format this

#         # loop through tilts
#         for tilt in dmpositions[1]:
            
#             piston = float(0)
#             tip = float(tip)
#             tilt = float(tilt)
#             dm.set_segment(segment, piston, tip, tilt)

#             data = getdata(apapane, box, dark)
            
#             scan[total] = data

#             # Append timestamp in UTC ISO format
#             timestamps.append(time.time())
            
#             time.sleep(0.1)

#             pbar.update()

#             total +=1

    
#     piston = float(0)
#     tip = float(0)
#     tilt = float(0)

#     plt.imshow(data)
#     plt.show()
    
#     dm.set_segment(segment, piston, tip, tilt) # return the segment back to normal
    
#     return scan, timestamps
            

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






if __name__ == '__main__':
    with open('scanparameters.json', 'r') as f:
        params = json.load(f)

    date = params['date']
    iteration = params['iteration']
    step_size = np.array(params['step_size'])
    scan_range = np.array(params['scan_range'])
    start_pos = np.array(params['start_pos'])
    peaks = params['peaks']
    boundingvals = params['boundingvals']
    box_halfwidth = params['box_halfwidth']
    iscred1 = params['iscred1']

    dark_filepath = str(glint_paths.DATA_ROOT / 'alignment_scans' / 'dark.fits')
    dark = getdark(dark_filepath)

    savepath = str(glint_paths.data_dir('alignment_scans', 'tiptiltscans', '2025', date, f'scan{iteration}'))
    checksavepath(savepath)

    # get the three photometry boxes
    boxes = [getbox(peak, boundingvals, box_halfwidth, iscred1) for peak in peaks]  

    nsteps = np.ceil(scan_range/step_size).astype(int) + 1
    dmpositions = [np.linspace(start_pos[i], start_pos[i] + scan_range[i], nsteps[i]) for i in range(2)]
    dmpositions = np.array(dmpositions)

    apapane, dm = open_devices()  

    excluded = [11, 20, 31]
    for segment in range(37):
        if segment in excluded:
            continue

        try:
            # tilt all excluded segs 
            tiltseg(dm, segments=excluded)

            scanned_cube, timestamps = scan(dm, dmpositions, apapane, dark, boxes, segment)

            primary_hdu = fits.PrimaryHDU()
            phot_hdu = fits.ImageHDU(data=scanned_cube, name='PHOTOMETRY')
            mount_hdu = fits.ImageHDU(data=dmpositions, name='DM_POS')
            timestamps_hdu = fits.ImageHDU(data=np.array(timestamps), name='TIMESTAMPS')

            hdul = fits.HDUList([primary_hdu, phot_hdu, mount_hdu, timestamps_hdu])
            filename = f'{savepath}/tiptiltscan_seg{segment}_{iteration}.fits'
            hdul.writeto(filename, overwrite=True)
            print(f'Saved {filename}')

        except Exception as e:
            import traceback
            traceback.print_exc()
            sys.exit(1)




