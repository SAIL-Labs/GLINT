import sys
sys.path.append('/home/scexao/glint/control-code/')

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
import subprocess
from datetime import datetime, timezone


PARAM_FILE = '/home/scexao/glint/alignment_scans/tiptiltscans/scanparameters.json'

def load_params():
    with open(PARAM_FILE, 'r') as f:
        return json.load(f)


def save_params(params):
    with open(PARAM_FILE, 'w') as f:
        json.dump(params, f, indent=2)

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


def tiltunused(dm):
    segments = [11, 20, 31]

    for seg in range(37):
        if seg not in segments:
            dm.set_segment(seg, 0, -5.5, -4)
        

def getdata(apapane, box, dark, nframes = 10) -> np.ndarray:
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


def scan(dm, dmpositions: np.ndarray, apapane: SHM, dark: np.ndarray, box: list, nframes) -> None:
    # Move dm to start position

    top, bottom, left, right = boxes[0]

    boxheight = bottom-top
    boxwidth = right-left

    nsteps = [len(dmpositions[0]), len(dmpositions[1])]  # Number of steps in the scan per axis
    totalsteps = nsteps[0]*nsteps[1]
    scan = np.zeros((totalsteps, boxheight, boxwidth)) 
    timestamps = []
    

    # scan = np.zeros((len(dmpositions[0]), len(dmpositions[1])))

    pbar = tqdm.tqdm(desc="Tip-tilt scan", total=totalsteps)

    # loop through tips
    total = 0
    for tip in dmpositions[0]:


        # print(f'tip: {tip}') # figure out how to format this

        # loop through tilts
        for tilt in dmpositions[1]:
            
            piston = float(0)
            tip = float(tip)
            tilt = float(tilt)
            dm.set_segment(segment, piston, tip, tilt)

            data = getdata(apapane, box, dark, nframes)
            
            scan[total] = data

            # Append timestamp in UTC ISO format
            timestamps.append(time.time())
            
            time.sleep(0.1)

            pbar.update()

            total +=1

    
    piston = float(0)
    tip = float(0)
    tilt = float(0)

    # plt.imshow(data)
    # plt.show()
    
    dm.set_segment(segment, piston, tip, tilt) # return the segment back to normal
    
    return scan, timestamps
            



def checksavepath(base_dir):

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



if __name__ == '__main__':
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
    base_dir = f'/home/scexao/glint/alignment_scans/tiptiltscans/{year}/{date}'

    # This may update iteration in the JSON
    savepath = checksavepath(base_dir)

    # Reload params AFTER checksavepath has possibly updated them
    params = load_params()
    iteration = params['iteration']

    # Now use refreshed params everywhere below
    date = params['date']
    nframes = params['nframes']
    step_size = np.array(params['step_size'])
    scan_range = np.array(params['scan_range'])
    start_pos = np.array(params['start_pos'])
    peaks = params['peaks']
    boundingvals = params['boundingvals']
    box_halfwidth = params['box_halfwidth']
    iscred1 = params['iscred1']


    # Get dark frame
    dark_filepath = '/home/scexao/glint/alignment_scans/dark.fits'
    dark = getdark(dark_filepath)

    

    # Get the spectral boxes: [top, bottom, left, right]
    boxes = [getbox(peak, boundingvals, box_halfwidth, iscred1) for peak in peaks]  


    for segment in [11, 20, 31]:
        # Spectra crop
        if segment == 11:
            box = boxes[0]
        elif segment == 20:
            box = boxes[1]
        elif segment == 31:
            box = boxes[2]
        else:
            print('Invalid segment')

        
        # Get the positions
        nsteps = np.ceil(scan_range/step_size).astype(int) + 1  # Number of steps in the scan (need to plus 1 to include the last position)
        dmpositions = [np.linspace(start_pos[i], start_pos[i] + scan_range[i], nsteps[i]) for i in range(2)]
        dmpositions = np.array(dmpositions)


        # Open devices
        apapane, dm = open_devices()  

        try:
            tiltunused(dm)
            scanned_img, timestamps = scan(dm, dmpositions, apapane, dark, box, nframes)

            # HDU 0: Primary (empty or summary)
            primary_hdu = fits.PrimaryHDU()

            # HDU 1: 3D photometry image cube
            phot_hdu = fits.ImageHDU(data=scanned_img, name='PHOTOMETRY')

            # HDU 2: mount positions (saved as an image or table)
            mount_hdu = fits.ImageHDU(data=dmpositions, name='DM_POS')

            timestamps_hdu = fits.ImageHDU(data=timestamps, name='TIMESTAMPS')

            hdul = fits.HDUList([primary_hdu, phot_hdu, mount_hdu, timestamps_hdu])

            filename = f'{savepath}/tiptiltscan_seg{segment}_{iteration}.fits'
            hdul.writeto(filename, overwrite=True)
            print(f'Saved {filename}')


            # # Save data
            # hdu = fits.PrimaryHDU(scanned_img)
            # hdul = fits.HDUList([hdu])
            # hdul.writeto(f'/home/scexao/glint/alignment_scans/tiptiltscans/2025/{date}/scan{iteration}/tiptilt_seg{segment}_{iteration}.fits', overwrite=True)


        except Exception as e:
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            sys.exit(1)


    # --- Run tip-tilt analysis automatically ---
    print("\nRunning tip-tilt analysis and updating parameter files...")

    analysis_script = "/home/scexao/glint/alignment_scans/tiptiltscans/read_tiptiltscan.py"

    try:
        subprocess.run(
            [
                "python",
                analysis_script,
                "--method",
                "elliptical_gaussian",
                "--update-param-file",
            ],
            check=True
        )
        print("Tip-tilt analysis complete and parameters updated.")

    except subprocess.CalledProcessError as e:
        print("Error running tip-tilt analysis script:")
        print(e)






# def tiltunused_api(mems: apiMEMsontrol.MEMS) -> None:
    # """
    # Tilt off all unused segments.

    # Parameters
    # ----------
    # mems : apiMEMsControl.MEMS
    #     MEMs object.

    # Returns
    # -------
    # None.
    # """

    # segments = [11,20,31]  # Used segments that should not be tilted away

    # try:
    #     for seg in range(37):
    #         if seg not in segments:
    #             err_code, minPiston, maxPiston = mems.get_segment_range(seg, 'piston', 0, 0, 0)
    #             if err_code:
    #                 raise Exception(mems.error_string(err_code))

    #             piston = int((minPiston + maxPiston)/2)

    #             err_code, minTilt, maxTilt = mems.get_segment_range(seg, 'tilt', piston, 0, 0)
    #             if err_code:
    #                 raise Exception(mems.error_string(err_code))

    #             mems.set_segment(seg, piston, 0.0, minTilt*1000)

    # except Exception as e:
    #     print(e)

    #     mems.closeDM()
    #     print('DM closed')

    #     sys.exit(1)



# def savedata(photometry_scans, iteration):
    """
    Save the photometry data.

    Parameters
    ----------
    photometry_scans : list
        List with the photometry arrays.
    select_axes : list
        List with the axes for this scan.
    iteration : int
        Iteration number.
    """

    # for i, photometry in enumerate(photometry_scans):
    #     hdu = fits.PrimaryHDU(photometry)
    #     hdul = fits.HDUList([hdu])
    #     hdul.writeto(f'tiptiltscan_spectra{i+1}_{iteration}.fits', overwrite=False)

