import numpy as np
import subprocess
import findhotspot  
from astropy.io import fits
from pyMilk.interfacing.shm import SHM



def find_newthetas(currentcoords, targetcoords, pertubcoords, pertubations, startthetas, step = 1e-7):

    
    """
        Calculates the next position for thetas. 
        In this case, [theta0, theta1, theta2, theta3] = [mirror1u, mirror1v, mirror2u, mirror2v].
        
        Parameters:
        currentcoords: Current psf coordinates.
            Format: np.array([x, y])

        targetcoords: Target psf coordinate. 
            Format: np.array([x, y])

        pertubcoords: Coordinates of hotspot after exam DOF undergoes a pertubation.
            Format: np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])

        pertubations: Amount the thetas are changed. 
            Format: np.array([dTheta0, dTheta1, dTheta2, dTheta3])

        startthetas: Starting position of of thetas before pertubation.
            Format: np.array([theta0, theta1, theta2, theta3])

        Returns:
        ndarray: new position for thetas
            Format: np.array([theta0, theta1, theta2, theta3])
    """
    

    
    # Find the current error (i.e. euclid distance between target and current psf), and the error after applying a small change in the mirror position.
    currenterror = np.linalg.norm(currentcoords - targetcoords)  # should look like: error
    pertuberror = np.linalg.norm(pertubcoords - targetcoords, axis = 1)  # should now look like [error0, error1, error2, error3]

    # Calculate the jacobian
    jacobian = (pertuberror - currenterror) / pertubations
    jacobianT = jacobian.T

    # Calculate the gradient
    grad = np.dot(jacobianT, currenterror)

    # Calculate the new thetas
    newthetas = startthetas - step * grad

    return newthetas

def get_jacobian(currentcoords, targetcoords, pertubcoords, pertubations):
    # Find the current error (i.e. euclid distance between target and current psf), and the error after applying a small change in the mirror position.
    currenterror = np.linalg.norm(currentcoords - targetcoords)  # should look like: error
    pertuberror = np.linalg.norm(pertubcoords - targetcoords, axis = 1)  # should now look like [error0, error1, error2, error3]

    # Calculate the jacobian
    jacobian = (pertuberror - currenterror) / pertubations
    jacobianT = jacobian.T

    return jacobianT

def movemirror(newthetas):

    commands = [
        f"glint_steering1 u goto {newthetas[0]}",
        f"glint_steering1 v goto {newthetas[1]}",
        f"glint_steering2 u goto {newthetas[2]}",
        f"glint_steering2 v goto {newthetas[3]}"
    ]
    
    for command in commands:
        subprocess.run(command, shell=True)

    return

def get_thetas():

    commands = [
        f"glint_steering1 u status",
        f"glint_steering1 v status",
        f"glint_steering2 u status",
        f"glint_steering2 v status"
    ]

    thetas = []
    for command in commands:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        thetas.append(float(result.stdout.split()[-1]))

    return thetas

def get_coords(brightframe= None, dark = None):
    """
    Get the current coordinates of the PSF.

    Returns:
    ndarray: Current coordinates of the PSF.
    """

    if brightframe is None:
        brightframe = SHM('apapane').get_data()
    if dark is None:
        dark = fits.getdata('darkpsf.fits')
        

    # Find the coordinates of the PSF
    coords = findhotspot.find_origin_com(brightframe - dark, threshold = 0.9)

    return coords


def get_pertubcoords(thetas, pertubation):

    # for each mirror:
    #   move mirror by pertubation
    #   get apapane frame
    #   get coords
    #   store coords in array
    #   return the mirror to original coords

    pertubcoords = []

    for i in range(4):
        # Move mirror
        movemirror(thetas[i] + pertubation[i])

        # Get coords
        pertubcoords.append(get_coords())

        # Return mirror
        movemirror(thetas)
    
    return pertubcoords



    

def mainloop():

    # Get target coorinates
    targetbright = fits.getdata('/frames/summitalignment/glint_focal_bright.fits')
    targetdark = fits.getdata('/frames/summitalignment/glint_focal_dark.fits')
    targetcoords = get_coords(targetbright, targetdark)
    
    # Get current thetas
    thetas = get_thetas()

    # define pertubation stepsize
    pertubation = [0.01, 0.01, 0.01, 0.01]

    step = 1e-7

    # Get current coords
    currentcoords = get_coords()

    # Get pertubed coords
    pertubcoords = get_pertubcoords()

    # make jacobian
    jacobian = get_jacobian(currentcoords, targetcoords, pertubcoords, pertubation)


    currenterror = np.linalg.norm(currentcoords - targetcoords)  # should look like: error

    # enter loop to calculate thetas
    while currenterror > 100:

        # Calculate the gradient
        grad = np.dot(jacobian, currenterror)

        # Calculate the new thetas
        newthetas = thetas - step * grad

        # update thetas 
        
        # move mirror

        # get current coord

        # calculate cuurent error





    # update current thetas













# def test_jacobian():
#     currentcoords = np.array([228.40098105160254, 399.3526400366813])
#     targetcoords = np.array([322.59722165059, 226.66784579359285])
#     pertubcoords = np.array([[227.41476587895832, 394.0029356441068], [228.71767470362875, 394.96928937031305], [227.73114250730856, 381.9280392194085], [203.69094047099549, 392.6364200224564]])

#     startthetas = np.array([0.293, 0.229, -0.019, -0.421])
#     endthetas = np.array([0.27, 0.21, -0.01, -0.41])
#     pertubations = endthetas - startthetas
#     print('pertubations:', pertubations)
#     jacobian(currentcoords, targetcoords, pertubcoords, pertubations, startthetas)


# if __name__ == '__main__':
#     test_jacobian()

#     # MAKE SURE THAT IT TESTS IF THERE IS A DARK