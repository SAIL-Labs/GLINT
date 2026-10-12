import chip
import numpy as np
import spectra
from PyQt5.QtWidgets import QGraphicsScene
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


def update_spectra(gui, pistons):

    n_apertures = 3
    n_phot = 3
    n_tricoupler = 3
    n_directional = 0
    coeff_photo = np.array([1/3, 1/3, 1/3])
    coeff_interf = np.array([1/2, 1/2, 1/2])

    pistons = pistons*1e-9

    figure = spectra.simulate_spectra(pistons, n_apertures, n_phot, n_tricoupler, n_directional, coeff_photo, coeff_interf)

    canvas = FigureCanvas(figure)
    scene = QGraphicsScene()
    gui.graphicsView_spectra.setScene(scene)
    scene.addWidget(canvas)


