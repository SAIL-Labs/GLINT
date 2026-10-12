import spectra
import numpy as np
from PyQt5.QtWidgets import QGraphicsScene
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtWidgets

class IntegerDelegate(QtWidgets.QStyledItemDelegate):

    def displayText(self, value, locale):
        try:
            # Ensure the display text shows only integer values
            return "{:.0f}".format(float(value))
        except ValueError:
            return value  # In case value cannot be converted, return as is

    def setModelData(self, editor, model, index):
        # Get the value from the editor and make sure it's stored as a float or integer
        value = editor.text()
        try:
            int_value = int(float(value))  # Convert to integer
            model.setData(index, int_value)  # Store as integer in the model
        except ValueError:
            model.setData(index, value)  # If conversion fails, store as string or raw

class FloatDelegate(QtWidgets.QStyledItemDelegate):

    def displayText(self, value, locale):
        try:
            # Ensure the display text shows only integer values
            return "{:.3f}".format(float(value))
        except ValueError:
            return value  # In case value cannot be converted, return as is

    def setModelData(self, editor, model, index):
        # Get the value from the editor and make sure it's stored as a float or integer
        value = editor.text()
        try:
            value =  float(value)  # Convert to integer
            model.setData(index, value)  # Store as integer in the model
        except ValueError:
            model.setData(index, value)  # If conversion fails, store as string or raw
            

def startup(gui):
    setup_spectra(gui)
    setdelegates(gui)


def setdelegates(gui):
    delegate = IntegerDelegate()
    gui.table_mountPos_rpy.setItemDelegate(delegate)
    gui.table_mountPos_xyz.setItemDelegate(delegate)

    delegate = FloatDelegate()
    gui.tab_DMposition_seg.setItemDelegate(delegate)
    gui.tab_DMposition_base.setItemDelegate(delegate)

def setup_spectra(gui):

    # Set up the initial figure
    num_apertures = 3

    pistons = np.zeros(num_apertures)
    figure = spectra.simulate_spectra(pistons)

    canvas = FigureCanvas(figure)
    scene = QGraphicsScene()
    gui.graphicsView_spectra.setScene(scene)
    scene.addWidget(canvas)



