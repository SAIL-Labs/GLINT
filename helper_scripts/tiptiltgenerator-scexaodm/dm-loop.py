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





def generate_wavefront(size=50, tilt_range=(-0.4, 0.4), superpose=False):
    """
    Generate a tip, tilt, or superposition wavefront enclosed in a circular mask.
    
    Parameters:
        size (int): Size of the square array (size x size).
        tilt_range (tuple): The range for tilt values (min, max).
        superpose (bool): If True, generate a random superposition of tip and tilt. 
                          If False, generate either tip or tilt randomly.

    Returns:
        np.ndarray: Generated wavefront array of type float32.
    """
    # Create a meshgrid for the coordinates
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # Generate random coefficients for tip and tilt
    tip_coeff = np.random.uniform(*tilt_range)
    tilt_coeff = np.random.uniform(*tilt_range)

    # Generate wavefront components
    tip_wavefront = tip_coeff * X
    tilt_wavefront = tilt_coeff * Y

    # Determine whether to superpose or generate one component
    if superpose:
        wavefront = tip_wavefront + tilt_wavefront
    else:
        if np.random.rand() > 0.5:
            wavefront = tip_wavefront
        else:
            wavefront = tilt_wavefront

    # Apply circular mask
    mask = R <= 1
    wavefront_circular = np.zeros_like(wavefront, dtype=np.float32)
    wavefront_circular[mask] = wavefront[mask]

    # Scale wavefront to the given range
    min_val, max_val = tilt_range
    wavefront_circular_scaled = min_val + (wavefront_circular - np.min(wavefront_circular)) * (max_val - min_val) / (np.max(wavefront_circular) - np.min(wavefront_circular))
    wavefront_circular_scaled = wavefront_circular_scaled.astype(np.float32)

    return wavefront_circular_scaled

# Example usage
if __name__ == "__main__":

    dm08 = SHM('dm00disp08') # Zernike coeeficient DM

    for i in range(30):
        # Generate a random superposition of tip and tilt
        wavefront = generate_wavefront(superpose=True)

        # Set the data
        dm08.set_data(wavefront)

        # Delay
        time.sleep(1)


    # # Save the generated wavefront to a text file
    # np.savetxt('random_wavefront_float32.txt', wavefront, fmt='%.6e')
    # print("Array saved to 'random_wavefront_float32.txt'")




