import numpy as np
import subprocess
import findhotspot  
from astropy.io import fits
from pyMilk.interfacing.shm import SHM
import time
import os
import matplotlib.pyplot as plt

# ============================
# --- Response Matrix IO ----
# ============================
def save_RM(R, filename="response_matrix.npy"):
    np.save(filename, R)
    print(f"Response matrix saved to {filename}.")

def load_RM(filename="response_matrix.npy"):
    if os.path.exists(filename):
        print(f"Loaded response matrix from {filename}.")
        return np.load(filename)
    else:
        print(f"Response matrix file {filename} not found.")
        return None



# ============================
# --- Main Loop -------------
# ============================
def mainloop(update_RM=False, RM_filename="response_matrix.npy"):
    """
    Aligns the PSF to a goal position using a feedback loop based on a response matrix.
    """
    # Load goal coordinates
    goal_bright = fits.getdata('/home/scexao/glint/psf_pupil_alignment/20251021/psf_zerovolts.fits')
    goal_dark = fits.getdata('/home/scexao/glint/psf_pupil_alignment/20251021/psf_dark.fits')
    goal_pos = get_hotspot(goal_bright, goal_dark)
    print(f"\nGoal pos: {goal_pos}\n")

    # Step 1: Load or calculate RM
    if update_RM or not os.path.exists(RM_filename):
        print("Calculating new response matrix...")
        perturbations = np.array([0.01, 0.01])
        R = calculate_RM(perturbations)
        save_RM(R, RM_filename)
    else:
        R = load_RM(RM_filename)

    # Step 2: Check invertibility
    print(f"Matrix R: {R}\n")
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        print("Warning: Response matrix is singular. Using pseudo-inverse.")
        R_inv = np.linalg.pinv(R)

    # Step 3: Setup loop parameters
    base_gain = 0.003     # A larger starting value
    min_gain = 0.0001    # Don't go below this
    gain = base_gain
    tolerance = 2       # pixel tolerance
    max_steps = 10
    sleep_time = 0.5

    last_offset = None
    initial_offset = None
    count = 0

    while count < max_steps:
        current_pos = get_hotspot()
        error = np.array(goal_pos) - np.array(current_pos)
        offset = np.linalg.norm(error)

        print(f"Step {count}: Current offset = {offset:.2f} pixels")

        if initial_offset is None:
            initial_offset = offset

        if offset < tolerance:
            print("Alignment achieved.")
            break

        # --- Adaptive gain control ---
        if last_offset is not None:
            improvement = np.abs((last_offset - offset) / last_offset)

            if improvement > 0.5:
                # If less than 5% improvement, reduce the gain
                gain *= 0.7
                gain = max(gain, min_gain)  # Clamp to minimum
                print(f"Large movement ({improvement:.2%}), reducing gain to {gain:.5f}")
            elif improvement < 0.3:
                # If less than 5% improvement, reduce the gain
                gain *= 1.2
                gain = max(gain, min_gain)  # Clamp to minimum
                print(f"Small movement ({improvement:.2%}), increasing gain to {gain:.5f}")

        print(f"Using gain: {gain:.5f}")

        # Calculate mirror move
        thetas = get_thetas()
        delta_thetas = gain * (R_inv @ error)
        new_thetas = thetas + delta_thetas

        print(f"Moving mirrors to: u = {new_thetas[0]:.4f}, v = {new_thetas[1]:.4f}\n")
 
        movemirror(new_thetas) # Won't run past this line is still moving mirror
        

        last_offset = offset
        count += 1
        # time.sleep(sleep_time). # Optional: add a small sleep to avoid overloading the system. 
  

    if count >= max_steps:
        print("Warning: Max steps reached without achieving alignment.")







def calculate_RM(pertubations):

    """
    Calculates the response matrix (RM) of the PSF position to mirror perturbations.
    
    Parameters:
    - perturbations: list or array of small u and v perturbations to apply to the mirror.

    Returns:
    - 2x2 response matrix (ndarray)
    """


    # R = np.array([[dx_du, dx_dv],              
    #               [dy_du, dy_dv]])

    current = get_hotspot()  # Initial PSF position
    pertubcoords = get_pertubcoords(pertubations) # Positions after pokes
    print(current)
    print(pertubcoords)

    # Convert to array and subtract original position to get deltas
    RespM = np.array(pertubcoords) - current

    return RespM


def get_pertubcoords(perturbations = [0.01, 0.01]):

    """
    Apply small perturbations to the mirror one axis at a time, 
    and record the PSF position for each.

    Returns:
    - list of PSF coordinates after each perturbation
    """

    perturbcoords = []
    thetas = get_thetas()
    n = len(thetas)

    print('Finding peturbed coordinates for perturbations:', perturbations)

    for i in range(n):
        
        # Create a new set of thetas with a perturbation on one axis
        perturbed = thetas.copy()
        perturbed[i] += perturbations[i]
        
        print(f'Moving theta {i} to {perturbed[i]}...')

        # Move mirror and record new hotspot
        movemirror(perturbed)
        time.sleep(1) # needs time to fix up its position and stabilise
        perturbcoords.append(get_hotspot())

        # Return mirror to original position
        print(f'Moving theta {i} back to {thetas[i]}...')
        movemirror(thetas)
    
    return perturbcoords


def movemirror(newthetas):
    """
    Send commands to move the mirror actuators to new positions.
    """

    commands = [
        f"glint_steering2 u goto {newthetas[0]}",
        f"glint_steering2 v goto {newthetas[1]}"
    ]
    
    for command in commands:
        subprocess.run(command, shell=True)

def get_thetas():
    """
    Queries the current mirror positions for axes u and v.

    Returns:
    - ndarray of two float values representing current thetas.
    """

    commands = [
        f"glint_steering2 u status",
        f"glint_steering2 v status"
    ]

    thetas = []
    for command in commands:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        try:
            thetas.append(float(result.stdout.split()[-1]))  # Might fail if output format changes
        except ValueError:
            print("Warning: Could not parse theta from:", result.stdout)
            thetas.append(0.0)

    return np.array(thetas)

def get_hotspot(brightframe= None, dark = None):
    """
    Gets the current hotspot (PSF) position from the detector frame.

    Returns:
    - x, y coordinate of the PSF.
    """

    start_x = 300
    start_y = 300
    end_x = 800
    end_y = 800 

    nframes = 50

    if brightframe is None:

        frames = []
        for i in range(nframes):
            frame = get_frame()
            frames.append(frame)
        
        brightframe = np.mean(np.array(frames), axis = 0)

    
    
    if dark is None:
        try:
            dark = fits.getdata('dark.fits')
        except Exception as e:
            print("Need to save a dark frame to current directory.")
        

    # Find the coordinates of the PSF
    img = brightframe.astype(float) - dark.astype(float)
    img = img/np.max(img)
    img = img[start_y:end_y,start_x:end_x]
    

    coords = findhotspot.find_origin_com(img, threshold = 0.8)
    # plt.imshow(img, origin='lower')
    # # plt.plot(coords[0], coords[1], 'rx')
    # plt.title('PSF centre: ({:.1f}, {:.1f})'.format(coords[0],coords[1]))
    # plt.show()

    return coords


def get_frame():
    """
    Returns the current frame from the shared memory.
    """
    
    
    return SHM('glintpg2').get_data()



    

if __name__ == '__main__':
    # To run mainloop, uncomment below



    # Set to True to recalculate and update the saved RM
    mainloop(update_RM=False)
    


# # def mainloop(update_RM=False, RM_filename="response_matrix.npy"):
#     """
#     Aligns the PSF to a goal position using a feedback loop based on a response matrix.
    
#     Parameters:
#     - update_RM (bool): If True, will recalculate and overwrite saved RM. If False, will use existing file if present.
#     - RM_filename (str): Path to saved RM file.
#     """

#     # Load goal coordinates
#     goal_bright = fits.getdata('./frames/summitalignment_03_03_25/psf_zerovolts_ir2.fits')
#     goal_dark = fits.getdata('./frames/summitalignment_03_03_25/psf_dark.fits')
#     goal_pos = get_hotspot(goal_bright, goal_dark)
#     print(f"\nGoal pos: {goal_pos}\n")

#     # Step 1: Load or calculate RM
#     if update_RM or not os.path.exists(RM_filename):
#         print("Calculating new response matrix...")
#         pertubations = np.array([0.01, 0.01])
#         R = calculate_RM(pertubations)
#         save_RM(R, RM_filename)
#     else:
#         R = load_RM(RM_filename)

#     # Check invertibility
#     print(f"Matrix R: {R}\n")
#     try:
#         R_inv = np.linalg.inv(R)
#     except np.linalg.LinAlgError:

#         print("Error: Response matrix is singular or ill-conditioned.")
#         R_inv = R
#         pass


#     # Step 2: Iterative loop
#     gain = [0.003, 0.02]
#     tolerance = 3  # pixels

#     # ls_thetas0 = []
#     # ls_thetas1 = []
#     # ls_offset = []

#     count = 0
#     while count <=20:
#         current_pos = get_hotspot()  # get x, y
#         error = np.array(goal_pos) - np.array(current_pos)   # x_goal - x, y_goal - y
        
#         print(f"Current pos: {current_pos}")
        
#         # offset = np.linalg.norm(error)
#         offset = np.sqrt((goal_pos[0]-current_pos[0])**2 + (goal_pos[1]-current_pos[1])**2)
#         # ls_offset.append(offset)

#         print(f"Offset: {offset}\n")

#         if offset < tolerance:
#             print("Alignment achieved.")
#             break

#         thetas = get_thetas()
#         # ls_thetas0.append(thetas[0])
#         # ls_thetas1.append(thetas[1])

#         print(f"Current mirror val: 2u = {thetas[0]}, 2v = {thetas[1]}")
#         delta_thetas = gain * R_inv @ error  # Matrix-vector multiplication
#         print(f"Delta thetas: {delta_thetas}")

#         new_thetas = thetas + delta_thetas  # this might have to be edited
#         print(f"New mirror val: 2u = {np.round(new_thetas[0], 3)}, 2v = {np.round(new_thetas[1], 3)}\n\n")


#         movemirror(new_thetas)  

#         # Optional: add a small sleep to avoid overloading the system
#         time.sleep(0.5)
#         count +=1
    
#     # plt.plot(ls_offset, ls_thetas0)
#     # plt.show()