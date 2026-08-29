from astropy.io import fits
import json

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit

def sin_squared_model(x, A, f, x0, C):
    return A * (1 - np.cos(2 * np.pi * f * (x - x0))) / 2 + C



wavelength = {10:'1650nm', 22:'1600nm', 35:'1550nm', 55:'1500nm'}
colours = {10:'C3', 22:'C2', 35:'C1', 55: 'C0'}
# 'C0', 'C2', 'C1', 'C3'
# opderror = {["11","31"]: 1, ["20","11"]:4, ["31","20"]: 10}

dark_iter = 13
date = '05-03'
year = '2026'


darkpath = f'/home/scexao/glint/alignment_scans/nullscans/{year}/{date}/scan{dark_iter}'

for iteration in [12]:

    for baseline in [["11","31"],["11","20"], ["20","31"]]: 

        if baseline == ["11","31"]:
            opderror = 1
            lowerbound = 0.5
            upperbound = 1.5
        elif baseline == ["11","20"]:
            opderror = 4
            lowerbound = -0.5
            upperbound = 0.5
        else:
            opderror = 10
            lowerbound = -0.5
            upperbound = 1
        plt.figure(figsize = (10,5))

        path = f'/home/scexao/glint/alignment_scans/nullscans/2026/{date}/scan{iteration}'

        # Open the file
        hdul = fits.open(f'{path}/avgmovie_{baseline[0]}:{baseline[1]}.fits')
        darkhdul = fits.open(f'{darkpath}/avgmovie_{baseline[0]}:{baseline[1]}.fits')

        movie = hdul['MOVIE'].data  # shape (11, 100, 4)
        opd = hdul['METADATA'].data['OPD']  # shape (11,)

        # darkmovie = darkhdul['MOVIE'].data  # shape (11, 100, 4)
        # darkopd = darkhdul['METADATA'].data['OPD']  # shape (11,)

        zoomed = True

        if not zoomed:
            lowerbound = -2
            upperbound = 2

        # print(darkmovie.shape)
        movie = movie[(opd>=lowerbound)&(opd<=upperbound)]
        # darkmovie = darkmovie[(opd>=lowerbound)&(opd<=upperbound)]
        # darkopd = darkopd[(opd>=lowerbound)&(opd<=upperbound)]
        opd = opd[(opd>=lowerbound)&(opd<=upperbound)]


        top = np.array([0])
        bottom = np.array([-1])

        for i in range (len(top)):
            summed_flux = movie[:, :, top[i]:bottom[i]].sum(axis=(1, 2))  # sum over height and width
            # darksummed_flux = darkmovie[:, :,top[i]:bottom[i]].sum(axis=(1, 2))  # sum over height and width
            
            maxsum = np.max(summed_flux)
            summed_flux_norm = summed_flux/maxsum
            # dark_flux_norm = darksummed_flux/maxsum

            # plt.plot(opd, summed_flux_norm, color = colours.get(top[i], "black"), label = wavelength[top[i]])
            try:
                wavelength_nm = 1550   # e.g., 1650
                wavelength_um = wavelength_nm / 1000
                initial_freq = 1 / wavelength_um



                p0 = [1, initial_freq, opderror, 0.1]
                popt, _ = curve_fit(sin_squared_model, opd, summed_flux_norm, p0=p0)

                # Extract fitted parameters
                A_fit, f_fit, x0_fit, C_fit = popt
                print(x0_fit)

                # Calculate trough positions
                n_values = np.arange(-10, 11)
                troughs = x0_fit + n_values / (2*f_fit)
                troughs_in_range = troughs[(troughs >= opd.min()) & (troughs <= opd.max())]

                # Plot trough lines
                for t in troughs_in_range:
                    plt.axvline(t, color=colours.get(top[i], "black"), linestyle=':', alpha=0.5)

                # Add red bold labels just under x-axis (without overlapping x-axis label)
                for t in troughs_in_range:
                    plt.text(t, -0.057, f'{t:.2f}', ha='center', va='top', fontsize=8,
                            color='blue', fontweight='bold', rotation=0,
                            transform=plt.gca().get_xaxis_transform())




                # Plot trough markers
                for t in troughs_in_range:
                    plt.axvline(t, color="blue", linestyle=':', alpha=0.5)


                opd_fit = np.linspace(opd.min(), opd.max(), 300)
                flux_fit = sin_squared_model(opd_fit, *popt)

                # get the y value of the min opd fit
                min_flux_fit = np.min(flux_fit)
                min_opd_fit = opd_fit[np.argmin(flux_fit)]
                max_flux_fit = np.max(flux_fit)
                max_opd_fit = opd_fit[np.argmax(flux_fit)]
                print(f"Minimum at OPD: {min_opd_fit:.2f} um, Flux: {min_flux_fit:.2f}")
                print(f"Maximum at OPD: {max_opd_fit:.2f} um, Flux: {max_flux_fit:.2f}")

                # get the value of teh flux fit for an opd of 2.83
                opd_value = min_opd_fit
                flux_value = sin_squared_model(opd_value, *popt)
                print(f"Flux at OPD {opd_value:.2f} um: {flux_value:.2f}")

                visibility = (max_flux_fit-flux_value)/(flux_value+max_flux_fit)
                print(f"Visibility: {visibility}")

                # plot a vline for the opd_value and the number of the visibility 
                plt.axvline(opd_value, color='orange', linestyle='--', label=f'OPD {opd_value:.2f} um. Visibility = {visibility:.3f}')


                plt.axvline(min_opd_fit, color='green', linestyle='--', label=f'Minimum at {min_opd_fit:.2f} um')
                plt.text(min_opd_fit, min_flux_fit, f'{min_flux_fit:.2f}', ha='center', va='bottom', fontsize=8,
                         color='green', fontweight='bold', rotation=0,
                         transform=plt.gca().get_xaxis_transform())

                plt.plot(opd, summed_flux_norm, 'o', markersize = 2, color='red', label='Data')
                plt.plot(opd_fit, flux_fit, '--', color="black", label = 'Fit')

            except RuntimeError:
                plt.plot(opd, summed_flux_norm, 'o-', color = colours.get(top[i], "black"), label = f"{wavelength[top[i]]} (fit failed)")
#

            # plt.plot(darkopd, dark_flux_norm,color = 'k', label='Detector noise')#colours.get(top[i], "black"))

        plt.xlabel('OPD (um)')
        plt.ylabel('Summed intensity')
        plt.title(f'Flux vs OPD: Baseline {baseline}')
        plt.grid(True)
        plt.legend()
        # plt.ylim([-0.05, 1.05])
        plt.savefig(f'{path}/fitted_nullscan_{baseline[0]}:{baseline[1]}_scan{iteration}.png')
        plt.show()

