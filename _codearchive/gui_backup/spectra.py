import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from chip import Chip



def simulate_spectra(x, n_apertures = 3, n_phot = 3, n_tricoupler = 3, n_directional = 0, coeff_photo = np.array([1/3, 1/3, 1/3]), coeff_interf = np.array([1/2, 1/2, 1/2])):
    
    n_outputs = n_phot + 3*n_tricoupler + 2*n_directional

    # H-band interferometry
    lambda_start = 1500e-9
    lambda_end = 1800e-9

    # Number of wavelengths to simulate: each pixel has a reolsution of about 30nm
    pixel_res = 3e-9

    # Number of wavelengths to simulate
    n = int((lambda_end-lambda_start)/pixel_res) 
    wavelengths = np.linspace(lambda_start, lambda_end, n)

    # Array that tracks the outputs of the chip for each wavelength: (12, n) array
    intensities = np.zeros((n_outputs, len(wavelengths)))

    # Amplitudes of the input waves
    amps = np.array([1, 1, 1])

    # Loop through each wavelength and calculate the output intensities
    for i, wavelength in enumerate(wavelengths):
        
        # Calculate the phase of each input wave
        phases = 2*np.pi*x/wavelength

        # Calculate the input waves
        input_wave = amps*np.exp(1j*phases)

        chip = Chip(input_wave, n_apertures, n_phot, n_tricoupler, n_directional, coeff_photo, coeff_interf)
        output_wave = chip.assemble()

        intensities[:,i] = np.abs(output_wave)**2  

    # Update figure to include the new intensities
    fig = Figure(figsize=(3, 4.7))
    axs = fig.subplots(12, sharex=True)
    wg_names = ['B1', 'N1', 'B2', 'B3', 'N2', 'B4', 'B5', 'N3', 'B6', 'P1', 'P2', 'P3']

    for j in range(n_outputs):
        axs[j].imshow([intensities[j]], cmap='viridis', aspect='auto', vmin = 0, vmax = 1)
        axs[j].get_yaxis().set_ticks([])
        axs[j].set_ylabel(wg_names[j], rotation=0, labelpad=20, fontsize=10, va='center')

    axs[-1].set_xticks(np.linspace(0,n, 4), np.linspace(lambda_start*1e9, lambda_end*1e9, 4, dtype=int))
    axs[-1].set_xlabel('Wavelength (nm)')

    return fig


