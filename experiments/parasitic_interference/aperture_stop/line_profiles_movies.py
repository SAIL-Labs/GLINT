import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# path = 'segment_11/movie_11:31.fits'
path = '/mnt/userdata/srossini/nullscans/2025/08-22/scan1/movie_31:11.fits'


with fits.open(path) as hdul:
    data = hdul[1].data  # Image stack
    opd = hdul[2].data    # OPDs


data = np.array(data)
opd = np.array(opd)
# Check the shape of the data
print(f"Data shape: {data.shape}")


data = np.mean(data[:,1:-1, 1:-1], axis = 1)
img = data


# plt.imshow(img[0], cmap='gray', vmin=0, vmax=1000)
# plt.show()

summed_data = np.sum(img[0], axis=1)
from scipy.signal import find_peaks
peaks, _ = find_peaks(summed_data, height=25000, distance=10)
# Plotting the peaks
plt.plot(summed_data, label='Summed Data')
plt.plot(peaks, summed_data[peaks], "x")
plt.title('Summed Data with Peaks')
plt.xlabel('Pixel Index')
plt.ylabel('Intensity')
plt.show()


# Make a box for a image for the specific spectral channel defined by a specific peak value that is 2px wide and 10px tall

def make_box(data, peak_index, width=4, height=2, start_y = 87):
    """
    Create a box around the peak in the data.
    
    Parameters:
    - data: 2D numpy array of the image data.
    - peak_index: Index of the peak in the summed data.
    - width: Width of the box (default is 2 pixels).
    - height: Height of the box (default is 10 pixels).
    
    Returns:
    - Boxed image data.
    """
    # Calculate the start and end indices for the box
    start_x = peak_index - width // 2
    end_x = peak_index + width // 2
    # start_y = 87
    end_y = start_y+height
    
    # Extract the box from the data
    return data[start_y:end_y, start_x:end_x]

# Create boxes for each peak
boxes = []
for peak in peaks:
    box = make_box(img[0], peak)
    boxes.append(box)
# Display the boxes
plt.figure(figsize=(2, 20))
for i, box in enumerate(boxes):
    plt.subplot(len(boxes), 1, i + 1)
    plt.imshow(box, cmap='gray', aspect='auto')
    plt.title(f'Box around Peak {i + 1}')
    plt.axis('off')
plt.tight_layout()
# plt.show()



# Now apply a box for each frame and summer over the box. Plot as a line plot on the y axis the summed value, and x axis is the frame number. Do this just for peak 0
def sum_boxed_data(data, peak_index, width=4, height=200):
    """
    Sum the data within a box around the peak for each frame.
    
    Parameters:
    - data: 3D numpy array of the image data (frames, height, width).
    - peak_index: Index of the peak in the summed data.
    - width: Width of the box (default is 2 pixels).
    - height: Height of the box (default is 10 pixels).
    
    Returns:
    - Summed values for each frame within the box.
    """
    summed_values = []
    for frame in data:
        box = make_box(frame, peak_index, width, height)
        summed_value = np.sum(box)
        summed_values.append(summed_value)
    return np.array(summed_values)
# Sum the boxed data for the first peak
peak_index = peaks[1]
summed_boxed_data = sum_boxed_data(img, peak_index)
# Plot the summed boxed data
plt.figure(figsize=(10, 6))
plt.plot(opd-np.min(opd), summed_boxed_data)#/np.max(summed_boxed_data))
# plt.title(f'Photometry: segment 11 (bottom)')
plt.xlabel('OPD (um)')
plt.ylabel('Normalised Summed Intensity')
# plt.ylim([0,1])
# plt.xlim([0,2.6])

plt.grid()
# plt.show()

print(np.max(summed_boxed_data))
print(np.min(summed_boxed_data[summed_boxed_data > 0]))



# Use make_box to plot a line pfile of peaK 0

# make a movie cycling through the images and plotting the box around the peak
import matplotlib.animation as animation

def update(frame):
    plt.clf()

    allboxes = []
    
    for peak in peaks:
        # Create a box around the peak for the current frame
        box = make_box(img[frame], peak, start_y=0, width=4, height=200)
        allboxes.append(box)
        
        
    # print(allboxes)
    # Sum the boxes together, then sum them along axis 1
    # summed_box = np.sum(np.array(allboxes[1:]), axis=0)

    # print(summed_box.shape)

    photbox = make_box(img[frame], peaks[-2], start_y=0, width=4, height=200)
    justphot = np.sum(np.array(photbox), axis=1)


    # for box in allboxes:
    #     plt.plot(np.sum(box, axis=1))
    # box = make_box(img[frame], peaks[2], start_y=0, width=4, height=200)

    # plt.plot(np.sum(summed_box, axis=1), label = 'Sum of tricouplers')
    plt.plot(justphot, label='Phot', color='orange')
    plt.title(f'OPD: {opd[frame]:.2f} um - Phot 20, but scanning baseline 11:31')
    plt.xlabel('Pixel Index')
    plt.ylabel('Summed Intensity')
    plt.ylim([0, 5000])  # Fixed y-axis range
    plt.legend()
    plt.grid()

fig = plt.figure(figsize=(10, 6))
ani = animation.FuncAnimation(fig, update, frames=img.shape[0], repeat=True)

# Save the animation with a faster frame rate
ani.save('phot20_scanning11:31.mp4', writer='ffmpeg', fps=20)
# Uncomment to plot the box around the peak

# for peak in [0,1,2]:

#     box = make_box(img[0], peaks[peak], start_y = 0, width=4, height=200)
#     plt.plot(np.sum(box, axis=1))
