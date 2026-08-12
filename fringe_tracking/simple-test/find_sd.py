import numpy as np
from astropy.io import fits
import os

# open the fits file
def getdata(file):
    return fits.getdata(file)

    
def sd(data):
    # Calculate the standard deviation of the data

    return np.std(data)


if __name__ == '__main__':
    path = './data/amp0.6'
    # Get the files in the current directory
    files = [f"{path}/{f}" for f in os.listdir(path)]

    dataarr = []

    # add data to an array
    for f in files:
        if f.endswith('.fits'):
            data = getdata(f)
            dataarr.append(data)


    # Calculate the standard deviation of the data
    std = sd(np.array(dataarr)/2)

    # Print the standard deviation
    print(f"Standard Deviation:", std)






