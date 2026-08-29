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

# Tip and tilt values fo segments [11,20,31] in that order



def getframe(apapane, nframes = 10) -> np.ndarray:
    """
    Takes data from the APAPANE camera, subtracts the dark frame, and crops the data to the spectral box.

    Parameters
    ----------
    apapane : SHM
        SHM object for the APAPANE camera.
    nframes : int, optional
        Number of frames to average over. The default is 100.
        
    Returns
    -------
    data : np.ndarray
        Data from the APAPANE camera.
    """

    # top, bottom, left, right = box
    bright = apapane.multi_recv_data(nframes) 
    bright = np.array(bright, dtype=float)  # Convert to float to avoid overflow

    return bright 




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
    
    data = avg - dark  # Subtract the dark frame
    data = data[top:bottom, left:right]  # Crop to the spectral box
    return data




def nullscan(dm, baseline: np.ndarray, tip: np.ndarray, tilt: np.ndarray, pistpositions, box, dark, nframes) -> None:

    seg1, seg2 = int(baseline[0]), int(baseline[1])
    tip_seg1, tip_seg2 = tip
    tilt_seg1, tilt_seg2 = tilt
    n = len(pistpositions)
    print(f'tip_seg1 = {tip_seg1}, tilt_seg1 = {tilt_seg1}')
    print(f'tip_seg2 = {tip_seg2}, tilt_seg2 = {tilt_seg2}')

    nullscan_seg1 = -1*np.zeros(n)
    nullscan_seg2 = -1*np.zeros(n - 1)
    tiltunused(dm)

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


    # height = box[1] - box[0] # for cred2
    # width = box[3] - box[2]
    timestamps = []

    pbar = tqdm.tqdm(desc="Null scan", total=2*n - 1)

   

    

    movie = np.zeros((2*n - 1, nframes, 256, 320))  # Store the data for each scan
    # 1. Loop over 2*n values.
    # 2. For the first n, keep segment 2 at its minimum piston value and scan segment 1 form maximum to minimum.
    # 3. Then at iteration n, segment 1 should be at a minimum position so move on to scanning over segment 2 from minimum to maximum
    for posnum in range (2*n - 1):

        # Scan 1 -----------------------------------------
        if posnum < n:
            frame = getdata(apapane, box, dark, nframes)
            fullframe = getframe(apapane, nframes)
            movie[posnum] = fullframe

            # New segment position
            newpos_seg1 = pos_seg1_scan1[posnum]
            dm.set_segment(seg1, newpos_seg1, tip_seg1, tilt_seg1)
            time.sleep(0.001)

            # Get data, sum over the spectral box, and store the summed flux
            summed_flux = np.sum(frame)
            nullscan_seg1[posnum] = summed_flux
            timestamps.append(time.time())

        # Scan 2 -----------------------------------------
        else:
            # break
            
            frame = getdata(apapane, box, dark, nframes)
            fullframe = getframe(apapane, nframes)
            movie[posnum] = fullframe
            posnum = posnum - n  # Reset posnum to start from 0 again when scanning over segment 2
            
            # New segment position
            newpos_seg2 = pos_seg2_scan2[posnum]
            dm.set_segment(seg2, newpos_seg2, tip_seg2, tilt_seg2)
            time.sleep(0.001)

            # Get data, sum over the spectral box, and store the summed flux
            summed_flux = np.sum(frame)
            nullscan_seg2[posnum] = summed_flux
            timestamps.append(time.time())
        
        pbar.update()


    # Store the summed data and the segment positions for each scan
    scan = np.concatenate([nullscan_seg1, nullscan_seg2])

    opd1 = pistpositions - pistpositions[-1]
    opd2 = np.abs(np.flip(pistpositions)[1:] - np.flip(pistpositions)[0])
    opd = np.concatenate([opd1, opd2])*2

    # save movie as fits file
    save_data(baseline, movie, opd, timestamps, savepath, avg = False)

    return movie, scan, opd, timestamps


    
def plot_scan(baseline:list, scan, opd, iteration:int, savepath, save:bool, avg:bool) -> None:

    # normscan = np.abs(scan)/np.max(scan)
    # plt.figure()
    plt.plot(opd, scan)
    plt.xlabel('OPD (um)')
    plt.ylabel('Summed flux')
    plt.title(f'Null scan for baseline {baseline}')
    plt.grid(True)
    if save:
        if avg:
            plt.savefig(f'{savepath}/avg_segment1_{seg1}:{seg2}_scan{iteration}.png')
            # plt.show()
        else:
            plt.savefig(f'{savepath}/segment1_{seg1}:{seg2}_scan{iteration}.png')
    
    plt.clf()
    
def plot_normscan(baseline:list, scan, opd, iteration:int, savepath, save:bool, avg:bool) -> None:

    normscan = np.abs(scan)/np.max(scan)
    # plt.figure()
    plt.plot(opd, normscan)
    plt.xlabel('OPD (um)')
    plt.ylabel('Normalised summed flux')
    plt.title(f'Null scan for baseline {baseline}')
    plt.grid(True)
    if save:
        if avg:
            plt.savefig(f'{savepath}/norm_avg_segment1_{seg1}:{seg2}_scan{iteration}.png')
            # plt.show()
        else:
            plt.savefig(f'{savepath}/norm_segment1_{seg1}:{seg2}_scan{iteration}.png')
    
    plt.clf()


def tiltunused(dm):
    segments = [20, 31]

    for seg in range(37):
        if seg not in segments:
            dm.set_segment(seg, 0, 0, 6)

# from astropy.io import fits

def save_data(baseline: list, movie, opd, timestamps, savepath, avg:bool) -> None:
    seg1, seg2 = baseline

    # Primary HDU (empty header or optional metadata)
    hdu0 = fits.PrimaryHDU()

    # Image HDU for the movie (multi-frame)
    hdu1 = fits.ImageHDU(movie, name='MOVIE')

    # Table HDU for OPD and Timestamps
    # col_opd = fits.Column(name='OPD', array=opd, format='E')          # 32-bit float
    # col_time = fits.Column(name='TIMESTAMP', array=timestamps, format='D')  # 64-bit float
    hdu2 = fits.ImageHDU(opd, name='OPD')

    # Write to file
    hdul = fits.HDUList([hdu0, hdu1, hdu2])

    if avg:
        hdul.writeto(f'{savepath}/avgmovie_{seg1}:{seg2}.fits', overwrite=True) # the timestamps will be from the last scan
    else:
        hdul.writeto(f'{savepath}/movie_{seg1}:{seg2}.fits', overwrite=True)

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
    # val1, val2 = boundingvals

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
        os.makedirs(savepath)
        print(f"Directory {savepath} created.\n")



if __name__ == "__main__":

    # Load settings
    with open('scanparameters.json', 'r') as f:
        params = json.load(f)

    # Now you can access:
    year = params['year']
    date = params['date']
    nframes = params['nframes']
    iteration = params['iteration']
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
    dark_filepath = '/home/scexao/glint/alignment_scans/darknull.fits'
    dark = getdark(dark_filepath)

    # savepath = f'/home/scexao/glint/alignment_scans/nullscans/2025/{date}/scan{iteration}'
    savepath = f'/mnt/userdata/srossini/nullscans/{year}/{date}/scan{iteration}/'
    checksavepath(savepath)

    nsteps = np.ceil(scan_range/step_size).astype(int) + 1  # Number of steps in the scan (need to plus 1 to include the last position)
    
    
    pistpositions = np.linspace(start_pos, start_pos + scan_range, nsteps)

    # Open the APAPANE camera and DM
    apapane, dm = open_devices()


    for baseline in [["20","31"]]:#, ["11","20"], ["20","31"]]:

        avgscan = []
        avgmovie = []

        for num in range(numavg):

            # Get the tip and tilt values for the segments
            seg1, seg2 = baseline
            tip = [tips[seg1], tips[seg2]]
            tilt = [tilts[seg1], tilts[seg2]]

            # Get the spectral box for the null scan, [top, bottom, left, right]
            box = getbox(baseline, nullpeaks, boundingvals, box_halfwidth, iscred1)

            # Perform the null scan
            movie, scan, opd, timestamps = nullscan(dm, baseline, tip, tilt, pistpositions, box, dark, nframes)
        
            # Plot and save the data
            # plot_scan(baseline, scan, opd, iteration, savepath,save = True, avg = False)
            # plot_normscan(baseline, scan, opd, iteration, savepath,save = True, avg = False)

            avgscan.append(scan)
            avgmovie.append(movie)

        avgscan = np.array(avgscan)
        print("scan shape before average:", avgscan.shape)
        avgscan = np.mean(avgscan, axis = 0)
        print("scan shape after average:", avgscan.shape)
        avgmovie = np.array(avgmovie)
        print("movie shape before average:", avgmovie.shape)
        avgmovie = np.mean(avgmovie, axis = 0)
        print("movie shape after average:", avgmovie.shape)



        # plt.imshow(avgmovie[0])
        # plt.show()
        plot_scan(baseline, avgscan, opd, iteration, savepath, save = True, avg = True)
        plot_normscan(baseline, avgscan, opd, iteration, savepath, save = True, avg = True)

        save_data(baseline, avgmovie, opd, timestamps, savepath, avg = True)


        

        # save_data(baseline, scan, opd, timestamps, iteration, savepath)
        # break

    # except Exception as e:
    #     print(e)
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     print(exc_type, fname, exc_tb.tb_lineno)
    #     sys.exit(1)







# def get_piston_range(dm, segment: int, tip: float, tilt: float) -> tuple:
#     """
#     Finds the piston range for a segment.

#     Parameters
#     ----------
#     dm : shmDMcontrol.DM
#         DM object.
#     segment : int
#         Segment to find the piston range for.
#     tip : float
#         Tip value for the segment.
#     tilt : float
#         Tilt value for the segment.

#     Returns
#     -------
#     minPiston : float
#         Minimum piston value.
#     maxPiston : float
#         Maximum piston value.
#     """

#     segment = int(segment)
#     tip = float(tip)
#     tilt = float(tilt)

#     err_code, minPiston, maxPiston = dm.get_segment_range(segment, 'piston', 0, tip, tilt)
#     if err_code:
#         raise Exception(dm.error_string(err_code))
#     return minPiston, maxPiston





