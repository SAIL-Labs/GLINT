'''
Goal: Move the SCExAO DM, save dm data array, save apapane frame, again.

Pseudocode:
1. Connect to shared memory
2. Define scexao DM 
3. Make an array of data to send
4. Move the DM
5. Delay
6. Repeat

'''
import numpy as np
from astropy.io import fits
from pyMilk.interfacing.shm import SHM  
import time

def generate_wavefront(size=50, tip_coeff=0.0, tilt_coeff=0.0):
    """
    Generate a tip, tilt, or superposition wavefront enclosed in a circular mask.
    
    Parameters:
        size (int): Size of the square array (size x size).
        tip_coeff (float): Coefficient for the tip component.
        tilt_coeff (float): Coefficient for the tilt component.

    Returns:
        np.ndarray: Generated wavefront array of type float32.
    """
    # Create a meshgrid for the coordinates
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # Generate wavefront components
    tip_wavefront = tip_coeff * X
    tilt_wavefront = tilt_coeff * Y

    # Superpose tip and tilt components
    wavefront = tip_wavefront + tilt_wavefront

    # Apply circular mask
    mask = R <= 1
    wavefront_circular = np.zeros_like(wavefront, dtype=np.float32)
    wavefront_circular[mask] = wavefront[mask]

    return wavefront_circular

def smooth_transition(start, end, steps):
    """
    Generate a smooth transition between start and end values over a given number of steps.
    
    Parameters:
        start (float): Starting value.
        end (float): Ending value.
        steps (int): Number of steps for the transition.

    Returns:
        np.ndarray: Array of values representing the smooth transition.
    """
    return np.linspace(start, end, steps)

# Example usage
if __name__ == "__main__":

    dm08 = SHM('dm00disp08') # Zernike coefficient DM
    steps = 100
    tilt_range = (-0.5, 0.5)

    # Initial random coefficients
    current_tip = np.random.uniform(*tilt_range)
    current_tilt = np.random.uniform(*tilt_range)

    for i in range(1000):
        # Next random coefficients
        next_tip = np.random.uniform(*tilt_range)
        next_tilt = np.random.uniform(*tilt_range)

        # Generate smooth transitions
        tip_transition = smooth_transition(current_tip, next_tip, steps)
        tilt_transition = smooth_transition(current_tilt, next_tilt, steps)

        for tip, tilt in zip(tip_transition, tilt_transition):
            # Generate wavefront for current tip and tilt
            wavefront = generate_wavefront(tip_coeff=tip, tilt_coeff=tilt)

            # Set the data
            dm08.set_data(wavefront)

            # Delay
            time.sleep(0.002)  # Adjust the delay as needed

        # Update current coefficients
        current_tip = next_tip
        current_tilt = next_tilt
