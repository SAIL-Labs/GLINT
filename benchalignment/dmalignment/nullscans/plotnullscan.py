from astropy.io import fits
import json

import matplotlib.pyplot as plt
import numpy as np


wavelength = {10:'1650nm', 22:'1600nm', 35:'1550nm', 55:'1500nm'}
colours = {10:'C3', 22:'C2', 35:'C1', 55: 'C0'}
# 'C0', 'C2', 'C1', 'C3'
date = '12-04'


darkpath = f'/home/scexao/glint/benchalignment/dmalignment/nullscans/2025/{date}/scan1'

for iteration in [2, 3, 4, 5, 6, 7, 8, 9, 10]:

    for baseline in [["31","20"]]:#["11","31"],["20","11"], ["31","20"]]: #,
        plt.figure(figsize = (10,5))

        path = f'/home/scexao/glint/benchalignment/dmalignment/nullscans/2025/{date}/scan{iteration}'

        # Open the file
        hdul = fits.open(f'{path}/avgmovie_{baseline[0]}:{baseline[1]}.fits')
        darkhdul = fits.open(f'{darkpath}/movie_{baseline[0]}:{baseline[1]}.fits')

        movie = hdul['MOVIE'].data  # shape (11, 100, 4)
        opd = hdul['METADATA'].data['OPD']  # shape (11,)

        darkmovie = darkhdul['MOVIE'].data  # shape (11, 100, 4)
        darkopd = darkhdul['METADATA'].data['OPD']  # shape (11,)

        # chromatic = 

        top = np.linspace(0,40, 10).astype(int)
     
        print(top)
        
        bottom = top + 2
        # print(bottom)
        print(movie.shape)
        for i in range(len(top)):
            # print(movie)
            summed_flux = movie[:,:,top[i]:bottom[i]].sum(axis=(1, 2))  # sum over height and width
            darksummed_flux = darkmovie[:,:,top[i]:bottom[i]].sum(axis=(1, 2))  # sum over height and width
            
            # Normalise summed_flux to its max value
            
            # summed_flux -= np.mean(darksummed_flux)
            # darksummed_flux -= np.mean(darksummed_flux)

            maxsum = np.max(summed_flux)
            # maxsum = 1

            summed_flux_norm = summed_flux#/maxsum
            dark_flux_norm = darksummed_flux#/maxsum

            plt.plot(opd, summed_flux_norm, color = colours.get(top[i], "blue"))#, label = wavelength[top[i]])
            # plt.plot(darkopd, dark_flux_norm,color = 'k', label='Detector noise')#colours.get(top[i], "black"))
            break

        plt.xlabel('OPD (um)')
        plt.ylabel('Summed intensity: narrowband')
        plt.title(f'Flux vs OPD: Aperture {baseline[0]}')
        plt.grid(True)
        # plt.legend()
        plt.ylim([1450, 2150])
        plt.savefig(f'{path}/nullscan_{baseline[0]}:{baseline[1]}_scan{iteration}.png')
        plt.show()

