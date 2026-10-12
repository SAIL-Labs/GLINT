import sys

from chipMountControl import Mount
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pyMilk.interfacing.shm import SHM
import glint_paths


def translationscan(apapane, axis, mountpositions, box, dark, nframes = 1):

    start = mountpositions[0]
    nsteps = len(mountpositions)

    # Move the axis to the start position
    mount.set_pos(axis, start)

    # Wait for the mount to stop moving and is at the correct position
    while mount.in_motion(axis) or mount.get_pos(axis) != start:
        pass

    # Scan the axis and save the data into the photometry array
    photometry_scan = np.zeros(nsteps)

    for i in range(nsteps):
        # Get data for each spectral box
        data = getdata(apapane, box, dark, nframes = 1) 

        # Sum the data and store in the photometry arrays
        photometry_scan[i] = np.sum(data)


        # Update the axis
        newpos = mountpositions[i]
        mount.set_pos(axis, newpos)

        # Wait for the mount to stop moving and is at the correct position
        while mount.in_motion(axis) or mount.get_pos(axis) != newpos:
            pass
    
    # new function that finds the peak of the photometry scan
    # try:
    #     peak_idx = findpeak(photometry_scan)
    # except Exception as e:
    #     print(e)
    #     peak_idx = 0

    try:
        peak_idx = fitgaussian(photometry_scan)
    except Exception as e:
        print(e)
        peak_idx = 0

    peakpos = mountpositions[peak_idx]
    print("peakpos:", peakpos)

    return peakpos

def getdata(apapane, box, dark, nframes = 1) -> np.ndarray:
    """
    Takes data from the APAPANE camera, subtracts the dark frame, and crops the data to the spectral box.

    Parameters
    ----------
    apapane : SHM
        SHM object for the APAPANE camera.
    box: list
        [top, bottom, left, right]
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
    
    avg = avg[3:-3, 3:-3]  # This crop is to remove the magic pixel
    data = avg - dark  # Subtract the dark frame
    data = data[top:bottom, left:right]  # Crop to the spectral box
    return data

def fitgaussian(photometry_scan):

    y = photometry_scan
    x = np.arange(len(y))

    # Fit the data to a gaussian
    popt, _ = curve_fit(gauss, x, y, p0=[1, np.argmax(y), 1])

    # plot the data and the gaussian
    plt.plot(photometry_scan)
    plt.plot(gauss(x, *popt))
    plt.show()


    # Find the x value of the peak of the gaussian
    print('Peak:', popt[1])

    peak = int(round(popt[1], 0))

    return peak

def findpeak(photometry_scan):
    '''
    Function to find the x index of the peak of a photometry scan.
    '''
    y = photometry_scan
    # x = mountpositions

    # Find the peaks
    peaks, _ = find_peaks(y, height=0)

    # plot the data and the gaussian
    plt.plot(photometry_scan)
    plt.plot(peaks, photometry_scan[peaks], "x")
    plt.show()

    # Get the index of the maximum peak out of y



    maxpeak = np.argmax(photometry_scan)



    return maxpeak

def gauss(x, a, b, c):
    '''
    Gaussian function.
    '''
    return a * np.exp(-(x - b)**2 / (2 * c**2))


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

def rasterscan(mount, dark, select_axes, mountpositions, box):
    
    # Setup ---------------------
    axlabel1, axlabel2 = select_axes  # Label for the axis e.g. x, y
    ax1, ax2 = 4, 6  # Axis number for the mount
    startpos1, startpos2 = mountpositions[0][0], mountpositions[1][0]  # Start position of the scan per axis
    nsteps = [len(mountpositions[0]), len(mountpositions[1])]  # Number of steps in the scan per axis
    photometry_scans = np.zeros(nsteps)  # photometry_scans as size (3 x nsteps) to store the photometry data i.e. 3 photometry channels





    
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
    for i in range(nsteps[0]):
        # Print the current position of the mount for the two axes to keep track of the scan progress
        print(f'{axlabel1}: {mount.get_pos(ax1)} \n {axlabel2}: {mount.get_pos(ax2)}')

        # Second axis
        for j in range(nsteps[1]):
            # Print the current position of the mount for the two axes to keep track of the scan progress
            # print(f'{axlabel1}: {mount.get_pos(ax1)} \n {axlabel2}: {mount.get_pos(ax2)}')

            # Get data for each spectral box
            data = getdata(apapane, box, dark, nframes = 1) 

            # Sum the data and store in the photometry arrays
            photometry_scans[i,j] = np.sum(data)

            
            # Update axis 2
            newpos2 = mountpositions[1,j]
            mount.set_pos(ax2, newpos2)

            # Wait for the mount to stop moving and is at the correct position
            while mount.in_motion(ax2) or mount.get_pos(ax2) != newpos2:
                pass

        # Update axis 1
        newpos1 = mountpositions[0,i]
        mount.set_pos(ax1, newpos1)

        # Return axis 2 to start position
        mount.set_pos(ax2, startpos2)

        # Wait for the mount to stop moving and is at the correct position
        while mount.in_motion(ax1) or mount.get_pos(ax1) != newpos1 or mount.in_motion(ax2) or mount.get_pos(ax2) != startpos2:
            pass
    

    return photometry_scans


def movemount(mount, axis, position):
    '''
    Function to move the mount to a position and wait for it to stop moving.
    '''
    mount.set_pos(axis, position)

    while mount.in_motion(axis) or mount.get_pos(axis) != position:
        pass

    return None


def scan(apapane, mount, dark, mountpositions, box):

    '''
    pseudocode:
        - Set up the scan array to be the size of (nframes, size of frame)
        - Move the mount to its start pitch/yaw position
        - Wait for the mount to stop moving
        - Set up a loop to scan in pitch/yaw
        - Translation scan
        - Save the frame
    '''
    xymountpos = mountpositions[0]
    pitchyawmountpos = mountpositions[1]
    pitches, yaws = pitchyawmountpos[0], pitchyawmountpos[1]
    x, y = xymountpos[0], xymountpos[1]

    nframes = len(pitches) * len(yaws)

    # Set up the scan array
    scan = np.zeros((nframes, len(x), len(y)))
    mountpos_ls = []

    # Move the mount to its start pitch/yaw position
    pitchax = 1
    yawax = 3


    i = 0

    for pitch in pitches:
        movemount(mount, pitchax, pitch)

        for yaw in yaws:
            print(f'pitch: {pitch}, yaw: {yaw}')

            movemount(mount, yawax, yaw)

            xaxis = 4
            yaxis = 6

            # translationscan(apapane, xaxis, x, box, dark)
            # translationscan(apapane, yaxis, y, box, dark)

            xyscan = rasterscan(mount, dark, ['x', 'y'], xymountpos, box)
            

            positions = [mount.get_pos(pitchax), mount.get_pos(yawax)]
            mountpos_ls.append(positions)

            # scan[i] = apapane.get_data()
            scan[i] = xyscan
            i += 1
            
    
    return scan, mountpos_ls



def getpositions(start_pos, scan_range, step_size):
    '''
    Function to get the positions for the mount to scan over.
    '''
    nsteps = np.ceil(scan_range/step_size).astype(int) + 1  # Number of steps in the scan (need to plus 1 to include the last position)
    positions = [np.linspace(start_pos[i], start_pos[i] + scan_range[i], nsteps[i]) for i in range(2)]
    positions = np.array(positions, dtype = int)

    return positions
    

if __name__ == '__main__':

    # Get dark frame
    dark_filepath = str(glint_paths.DATA_ROOT / '_dataarchive' / 'apapane' / 'darks' / 'apapane_2024-09-09_04:12:26.899782.fits')
    dark = getdark(dark_filepath)

    # Open devices
    apapane, mount = open_devices()  
    iteration = 8

    # Apapane spectral boxes
    peaks = [245,225,206]  # Centre of the spectral boxes
    boundingvals = [200,260]  # Bounding the length of the spectral boxes
    box_halfwidth = 2  # Halfwidth of the spectral box to sum over
    iscred1 = True   # If True, the spectra are horizontal, if False, the spectra are vertical 

    # Get the spectral boxes: [top, bottom, left, right]
    boxes = [getbox(peak, boundingvals, box_halfwidth, iscred1) for peak in peaks] 
    spectralbox = boxes[1]  # Spectral box to scan over 

    # Set up the xy scan -----
    select_axes = ['x','y']
    step_size = np.array([15, 15])
    scan_range = np.array([150, 150]) 
    start_pos = np.array([3050, 3200])

    # movemount(mount, 4, 3115)
    # movemount(mount, 6, 3295)

    movemount(mount, 4, 3140)
    movemount(mount, 6, 3285)



    # Get the positions for the mount
    xypositions = getpositions(start_pos, scan_range, step_size)


    # Set up the pitch/yaw scan -----
    select_axes = ['pitch','yaw']
    scan_range = np.array([1000, 1000])
    step_size = np.array([100, 100])
    start_pos = np.array([-400, -400])

    # Get the positions for the mount
    pitchyawpositions = getpositions(start_pos, scan_range, step_size)


    mountpositions = [xypositions, pitchyawpositions]


    # Do the scan
    scan, mountpos_ls = scan(apapane, mount, dark, mountpositions, spectralbox)

    hdu = fits.PrimaryHDU(scan)
    hdul = fits.HDUList([hdu])
    hdul.writeto(str(glint_paths.data_dir('_dataarchive', 'wiggles', f'scan{iteration}') / f'scan{iteration}.fits'), overwrite = True)

    hdu = fits.PrimaryHDU(mountpos_ls)
    hdul = fits.HDUList([hdu])
    hdul.writeto(str(glint_paths.data_dir('_dataarchive', 'wiggles', f'scan{iteration}') / f'mountpos{iteration}.fits'), overwrite = True)

    print('scan complete')




    # box = boxes[0]
    # axis = 4
    # mountpos = xypositions[0]

    # # Do the scan
    # peakpos, peakposgauss = translationscan(apapane, axis, mountpos, box, box_halfwidth, dark, nframes = 1)
    
    # mount.set_pos(axis, peakpos)

    # while mount.in_motion(axis) or mount.get_pos(axis) != peakpos:
    #     pass

    # axis = 6
    # mountpos = xypositions[1]

    # peakpos, peakposgauss = translationscan(apapane, axis, mountpos, box, box_halfwidth, dark, nframes = 1)

    # mount.set_pos(axis, peakpos)

    # while mount.in_motion(axis) or mount.get_pos(axis) != peakpos:
    #     pass

    # print('Scan complete')

    
