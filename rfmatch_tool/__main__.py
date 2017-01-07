import sys
import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore
from PyQt5 import QtGui, QtWidgets

# Plotting libraty
import pyqtgraph as pg
from pyqtgraph.dockarea import *

from gui.main_window import Ui_MainWindow
from components.Circuit import Circuit

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Set the circuit (contains all data)
        self.circuit = Circuit()
        self.plot_docks = {}

        # GUI stuff
        self.setupUi(self)

        # Connect the circuit list
        self.lvComponents.setModel(self.circuit)
#        self.listView.setIconSize(QSize(125,150))
        self.lvComponents.setIconSize(QtCore.QSize(75,90))

        # Add a dock widget
        self.test_dock_widget()

        # Update all gui elements with the datamodel
        self.update_gui_data()

    def update_gui_data(self):
        """
        Update all the combo-boxes, list views, plots, etc with the current circuit data
        :return:
        """
        # Set the parameters (S,Y,Z, ...)
        self.cbParameter.clear()
        self.cbParameter.insertItems(0, self.circuit.parameter_list)

    def test_dock_widget(self):
        # d1 = Dock("Dock1", size=(1, 1), closable=True)  ## give this dock the minimum possible size
        # d1.sigClosed.connect(self.dock_close_event)
        # d2 = Dock("Dock2 - Console", size=(500, 300), closable=True)
        # d2.sigClosed.connect(self.dock_close_event)
        # d3 = Dock("Dock3", size=(500, 400), closable=True)
        # d3.sigClosed.connect(self.dock_close_event)
        # d4 = Dock("Dock4 (tabbed) - Plot", size=(500, 200), closable=True)
        # d4.sigClosed.connect(self.dock_close_event)
        # d5 = Dock("Dock5 - Image", size=(500, 200), closable=True)
        # d5.sigClosed.connect(self.dock_close_event)
        # d6 = Dock("Dock6 (tabbed) - Plot", size=(500, 200), closable=True)
        # d6.sigClosed.connect(self.dock_close_event)
        # self.dock_area.addDock(d1, 'left')  ## place d1 at left edge of dock area (it will fill
        # self.dock_area.addDock(d2, 'right')  ## place d2 at right edge of dock area
        # self.dock_area.addDock(d3, 'bottom', d1)  ## place d3 at bottom edge of d1
        # self.dock_area.addDock(d4, 'right')  ## place d4 at right edge of dock area
        # self.dock_area.addDock(d5, 'left', d1)  ## place d5 at left edge of d1
        # self.dock_area.addDock(d6, 'top', d4)  ## place d5 at top edge of d4
        #
        # ## Test ability to move docks programatically after they have been placed
        # self.dock_area.moveDock(d4, 'top', d2)  ## move d4 to top edge of d2
        # self.dock_area.moveDock(d6, 'above', d4)  ## move d6 to stack on top of d4
        # self.dock_area.moveDock(d5, 'top', d2)  ## move d5 to top edge of d2

        # Setup the data
        freq = np.logspace(0, 6, 51)
        self.circuit.circuit.calc(freq)
        s11 = np.abs(self.circuit.circuit.s11[None])
        for n in range(4):
            self.add_dock_plot("test{}".format(n), s11)

    #######################
    # Plot dock functions #
    #######################
    def dock_close_event(self, dock_calling):
        if hash(dock_calling) in self.plot_docks:
            print('Dock with ID {} closed'.format(dock_calling.id))
        else:
            print('Unknown dock closed with ID {}'.format(dock_calling.id))

    def add_dock_plot(self, name, data):
        dock = PlotDock(data, name, closable=True)
        # Keep track of the dock in the plot lists (__hash__ is used to calculate the key)
        self.plot_docks[hash(dock)] = dock

        self.dock_area.addDock(dock, 'right')    # Place dock to the right
        dock.sigClosed.connect(self.dock_close_event)

class PlotDock(Dock):
    _ID = 0
    def __init__(self, data=None, *args, **kwargs):
        super(PlotDock, self).__init__(*args, **kwargs)
        # Keep track of the ID (give each dock an unique ID)
        self.__id = self._ID
        self.__class__._ID += 1

        # Setup the Plot + data
        self.data = data
        self.plot_widget = pg.PlotWidget(title="Z11")
        self.addWidget(self.plot_widget)
        self.plot()

    def plot(self):
        x = self.data.index
        y = self.data.values
        self.plot_widget.plot(x, y)

    @property
    def id(self):
        return self.__id

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        try:
            return self.id == other.id
        except:
            return False

    def __ne__(self, other):
        return not(self == other)

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
