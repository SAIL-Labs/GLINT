from astropy.io import fits
import json

import matplotlib.pyplot as plt
import numpy as np


# wavelength = {10:'1650nm', 22:'1600nm', 35:'1550nm', 55:'1500nm'}
wavelength = {0:'1650nm', 10:'1600nm', 15:'1550nm', 20:'1500nm', 25:'ignore'}
# colours = {10:'C3', 22:'C2', 35:'C1', 55: 'C0'}
colours = {0:'C3', 10:'C2', 15:'C1', 20: 'C0', 25: 'C4'}
# 'C0', 'C2', 'C1', 'C3'
date = '06-25'


darkpath = f'/home/scexao/glint/benchalignment/dmalignment/nullscans/2026/{date}/scan2'

for iteration in [1]:

    for baseline in [["11","31"],["11","20"]]:#, ["20","31"]]: #,
        plt.figure(figsize = (10,5))

        path = f'/home/scexao/glint/benchalignment/dmalignment/nullscans/2026/{date}/scan{iteration}'

        # Open the file
        hdul = fits.open(f'{path}/avgmovie_{baseline[0]}:{baseline[1]}.fits')
        # darkhdul = fits.open(f'{darkpath}/avgmovie_{baseline[0]}:{baseline[1]}.fits')

        movie = hdul['MOVIE'].data  # shape (11, 100, 4)
        opd = hdul['METADATA'].data['OPD']  # shape (11,)

        # darkmovie = darkhdul['MOVIE'].data  # shape (11, 100, 4)
        # darkopd = darkhdul['METADATA'].data['OPD']  # shape (11,)

        # chromatic = 

        top = np.array([0, 10, 15, 20, 25]) 
        # top = np.array([10, 22, 35, 55]) 
        
        bottom = top + 2
        for i in range (len(top)):
            summed_flux = movie[:, :, top[i]:bottom[i]].sum(axis=(1, 2))  # sum over height and width
            # darksummed_flux = darkmovie[:, :, top[i]:bottom[i]].sum(axis=(1, 2))  # sum over height and width
            
            # Normalise summed_flux to its max value
            
            # summed_flux -= np.mean(darksummed_flux)
            # darksummed_flux -= np.mean(darksummed_flux)

            maxsum = np.max(summed_flux)

            summed_flux_norm = summed_flux/maxsum
            # dark_flux_norm = darksummed_flux/maxsum

            plt.plot(opd, summed_flux_norm, color = colours.get(top[i]), label = wavelength[top[i]])
            # plt.plot(darkopd, dark_flux_norm,color = 'k')#, label='Detector noise')#colours.get(top[i], "black"))

        plt.xlabel('OPD (um)')
        plt.ylabel('Normalised summed intensity')
        plt.title(f'Flux vs OPD: Baseline {baseline}')
        plt.grid(True)
        plt.legend()
        # plt.ylim([-0.05, 1.05])
        plt.savefig(f'{path}/unnorm_nullscan_{baseline[0]}:{baseline[1]}_scan{iteration}.png')
        plt.show()

