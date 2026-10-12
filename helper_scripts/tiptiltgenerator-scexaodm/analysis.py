from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


with fits.open('test1/dark.fits') as hdul:
    dark = hdul[0].data.astype(np.float32)
    dark = np.mean(dark, axis = 0)



nums = np.arange(10)+1
bright1 = []
null = []
bright2 = []
for n in nums:
    with fits.open(f'test1/pos{n}.fits') as hdul:
        data = hdul[0].data.astype(np.float32)
        data = np.mean(data, axis = 0) - dark

    bright1.append(np.sum(data[179:181, 263:267]))
    null.append(np.sum(data[179:181, 293:297]))
    bright2.append(np.sum(data[179:181, 323:327]))


plt.plot(bright1, label = 'bright1')
plt.plot(null, label = 'null')
plt.plot(bright2, label = 'bright2')
plt.legend()
plt.show()