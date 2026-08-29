import sys
sys.path.append('/home/scexao/glint/hardwarescripts/')

from chipMountControl import Mount
from pyMilk.interfacing.shm import SHM
import numpy as np
from astropy.io import fits
import json

AXES = {'pitch':1, 'roll':2, 'yaw':3, 'x':4, 'z':5, 'y':6}


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
        dark = hdul[0].data  # This crop is to remove the magic pixel
        dark = np.array(np.mean(dark, axis=0), dtype=float)  # Convert to float to avoid overflow

    return dark

def open_devices():
    """
    Open the APAPANE camera and MEMS.

    Returns
    -------
    apapane : SHM
        SHM object for the APAPANE camera.
    mems : apiMEMsControl.MEMS
        MEMs object.
    """

    apapane = SHM('apapane')
    mount = Mount('/dev/serial/by-id/usb-SURUGA_SEIKI_SURUGA_SEIKI_DS102-if00-port0', 38400)

    return apapane, mount


def getdata(apapane, box,  box_halfwidth, dark, nframes = 1) -> np.ndarray:
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
        Number of frames to average over. The default is 1.

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



def savedata(path, photometry_scans, mount_positions, select_axes, iteration):
    """
    Save the photometry data (as image cubes) and mount positions in a multi-extension FITS file.

    Parameters
    ----------
    path : str
        Path to save the FITS file.
    photometry_scans : list or ndarray
        List/array of shape (3, nsteps, boxheight, boxwidth).
    mount_positions : ndarray
        Array of shape (nsteps, 2), with scan positions.
    select_axes : list
        Axes being scanned, e.g. ['x', 'y'].
    iteration : int
        Iteration number.
    """

    axlabel1, axlabel2 = select_axes

    for i, photometry in enumerate(photometry_scans):
        # HDU 0: Primary (empty or summary)
        primary_hdu = fits.PrimaryHDU()

        # HDU 1: 3D photometry image cube
        phot_hdu = fits.ImageHDU(data=photometry, name='PHOTOMETRY')

        # HDU 2: mount positions (saved as an image or table)
        mount_hdu = fits.ImageHDU(data=mount_positions, name='MOUNT_POS')

        hdul = fits.HDUList([primary_hdu, phot_hdu, mount_hdu])

        filename = f'{path}/{axlabel1}{axlabel2}scan_spectra{i+1}_{iteration}.fits'
        hdul.writeto(filename, overwrite=True)
        print(f'Saved {filename}')



def rasterscan(mount, dark, select_axes, mountpositions, boxes, box_halfwidth):

    top, bottom, left, right = boxes[0]

    boxheight = bottom-top
    boxwidth = right-left
    

    
    # Setup ---------------------
    axlabel1, axlabel2 = select_axes  # Label for the axis e.g. x, y
    ax1, ax2 = AXES[axlabel1], AXES[axlabel2]  # Axis number for the mount
    startpos1, startpos2 = mountpositions[0][0], mountpositions[1][0]  # Start position of the scan per axis
    nsteps = [len(mountpositions[0]), len(mountpositions[1])]  # Number of steps in the scan per axis
    totalsteps = nsteps[0]*nsteps[1]
    photometry_scans = np.array([np.zeros((totalsteps, boxheight, boxwidth)) for _ in range(3)]) # photometry_scans as size (3 x nsteps) to store the photometry data i.e. 3 photometry channels




    
    # Move mount to start position ---------------------
    print(f'Axis {axlabel1} position before moving: {mount.get_pos(ax1)}')
    mount.set_pos(ax1, startpos1) 
    while mount.in_motion(ax1) or mount.get_pos(ax1) != startpos1:
        pass
    print(f'Axis {axlabel1} scan start position: {mount.get_pos(ax1)}')

    print(f'Axis {axlabel2} position before moving: {mount.get_pos(ax2)}')
    mount.set_pos(ax2, startpos2) 
    while mount.in_motion(ax2) or mount.get_pos(ax2) != startpos2:
        pass
    print(f'Axis {axlabel2} scan start position: {mount.get_pos(ax2)}')
        

        




    # Raster scan ---------------------
    # First axis 
    step = 0
    for i in range(nsteps[0]):
        # Print the current position of the mount for the two axes to keep track of the scan progress
        print(f'{axlabel1}: {mount.get_pos(ax1)} \n {axlabel2}: {mount.get_pos(ax2)}')

        # Second axis
        for j in range(nsteps[1]):
            # Print the current position of the mount for the two axes to keep track of the scan progress
            # print(f'{axlabel1}: {mount.get_pos(ax1)} \n {axlabel2}: {mount.get_pos(ax2)}')

            # Get data for each spectral box
            data1, data2, data3 = [getdata(apapane, box, box_halfwidth, dark, nframes = 1) for box in boxes]

            # Sum the data and store in the photometry arrays
            photometry_scans[0][step] = data1
            photometry_scans[1][step] = data2
            photometry_scans[2][step] = data3

            
            # Update axis 2
            newpos2 = mountpositions[1,j]
            mount.set_pos(ax2, newpos2)

            # Wait for the mount to stop moving and is at the correct position
            while mount.in_motion(ax2) or mount.get_pos(ax2) != newpos2:
                pass
        
            step+=1

        # Update axis 1
        newpos1 = mountpositions[0,i]
        mount.set_pos(ax1, newpos1)

        # Return axis 2 to start position
        mount.set_pos(ax2, startpos2)

        # Wait for the mount to stop moving and is at the correct position
        while mount.in_motion(ax1) or mount.get_pos(ax1) != newpos1 or mount.in_motion(ax2) or mount.get_pos(ax2) != startpos2:
            pass
    

    return photometry_scans

def checksavepath(savepath):
    # i want to make sure it doesnt already exist, and if it does, increase the iteration number saved in the json file. if it doesn exist, make a new directory for it
    import os
    if os.path.isdir(savepath):
        # check if it is empty

        if os.listdir(savepath):
            print("Directory already exists and is not empty!")
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
    select_axes = params['select_axes']
    step_size = np.array(params['step_size'])
    scan_range = np.array(params['scan_range'])
    start_pos = np.array(params['start_pos'])
    peaks = params['peaks']
    boundingvals = params['boundingvals']
    box_halfwidth = params['box_halfwidth']
    iscred1 = params['iscred1']
    rollstart = params['rollstart']
    rollrange = params['rollrange']
    rollstepsize = params['rollstepsize']


    # Get dark frame
    dark_filepath = '/home/scexao/glint/benchalignment/mountalignment/dark.fits'
    dark = getdark(dark_filepath)

    

    # Get dark frame
    dark_filepath = '/home/scexao/glint/benchalignment/mountalignment/dark.fits'
    dark = getdark(dark_filepath)

    # Open devices
    apapane, mount = open_devices()  

    
    # Get the spectral boxes: [top, bottom, left, right]
    boxes = [getbox(peak, boundingvals, box_halfwidth, iscred1) for peak in peaks]  

    # Get the positions for the mount
    nsteps = np.ceil(scan_range/step_size).astype(int) + 1  # Number of steps in the scan (need to plus 1 to include the last position)
    mountpositions = [np.linspace(start_pos[i], start_pos[i] + scan_range[i], nsteps[i]) for i in range(2)]
    mountpositions = np.array(mountpositions, dtype = int)

    rollpositions = np.linspace(rollstart, rollstart + rollrange, int(rollrange/rollstepsize) + 1)
    rollpositions = np.array(rollpositions, dtype=int)
    for rollpos in rollpositions:
        mount.set_pos(2, rollpos)
        while mount.in_motion(2) or mount.get_pos(2) != rollpos:
            pass    
        print(f'Roll position set to {mount.get_pos(2)}')


        savepath = f'/home/scexao/glint/benchalignment/mountalignment/rollscans/2025/{date}/scan{iteration}/roll_{rollpos}'
        checksavepath(savepath)
        
        # Do the scan
        photometry_scans = rasterscan(mount, dark, select_axes, mountpositions, boxes, box_halfwidth)

        # Save the scan data
        savedata(savepath, photometry_scans, mountpositions, select_axes, iteration)



    
