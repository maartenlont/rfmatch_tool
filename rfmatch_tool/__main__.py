import sys
import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore
#from PyQt5 import QtGui, QtWidgets

from gui.main_window import Ui_MainWindow
from gui.PlotDock import PlotDock, Modifier
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
        self.lvComponents.setIconSize(QtCore.QSize(75,90))

        # Add a dock widget
        self.dock_area.temporary=False      # Dock area is not temporary (fixes bug when the last dock is closed)
#        self.default_dock_setup()

        # Update all gui elements with the datamodel
        self.update_gui_data()

        # Connect buttons
        self.pbPlot.clicked.connect(self.add_plot)
        self.pbAddPoints.clicked.connect(self.add_points)

        self.tbCircMin.clicked.connect(self.circ_remove)

    def update_gui_data(self):
        """
        Update all the combo-boxes, list views, plots, etc with the current circuit data
        :return:
        """
        # Set the parameters (S,Y,Z, ...)
        self.cbParameter.clear()
        self.cbParameter.insertItems(0, self.circuit.parameter_list)

        # Set all possible modifiers
        self.cbModifier.clear()
        self.cbModifier.insertItems(0, Modifier.__members__.keys())

    def add_plot(self):
        # Get modifier
        modifier = Modifier[self.cbModifier.currentText()]
        # Get parameter
        parameter = self.cbParameter.currentText()
        # Plot only data of the component:
        # data = getattr(self.circuit.circuit, parameter, None)
        data = getattr(self.circuit.circuit, parameter, None)

        self.add_dock_plot(data=data, modifier=modifier)

    def add_points(self):
        x_start = float(self.sbXstart.value())
        x_stop = float(self.sbXstop.value())
        x_num = int(self.sbXnum.value())

        if self.cbXtype.currentText() == 'Lin':
            freq = np.linspace(x_start, x_stop, x_num)
        else:
            freq = np.logspace(np.log10(x_start), np.log10(x_stop), x_num)

        self.circuit.circuit.calc(freq)

        self.dock_update()

    ######################################
    # Add/Remove/Edit circuit components #
    ######################################
    def circ_remove(self):
        # Get selected circuit elements (only one can be selected)
        selected = self.lvComponents.selectedIndexes()

        if selected[0].row() > 0: # Do not remove the load (index=0)
            # Remove it through the model
            self.circuit.remove_component(selected[0])

            # Update all the plots
            self.dock_update()

    #######################
    # Plot dock functions #
    #######################
    def default_dock_setup(self):
        # Setup the data
        freq = np.logspace(0, 6, 11)
        self.circuit.circuit.calc(freq)
        d11 = self.add_dock_plot(self.circuit.circuit.s11)
        d12 = self.add_dock_plot(self.circuit.circuit.s12)
        d21 = self.add_dock_plot(self.circuit.circuit.s21)
        d22 = self.add_dock_plot(self.circuit.circuit.s22)
        # Place the s parameters in logic order
        self.dock_area.moveDock(d12, 'right', d11)
        self.dock_area.moveDock(d21, 'bottom', d11)
        self.dock_area.moveDock(d22, 'bottom', d12)

    def dock_update(self):
        for key, dock in self.plot_docks.items():
            # TODO: Be able to plot based on the selected elements
            # For now plot the entire circuit (circuit.circuit)
            parameter = dock.data.varname
            data = getattr(self.circuit.circuit, parameter, None)

            dock.update(data=data)

    def dock_close_event(self, dock_calling):
        if hash(dock_calling) in self.plot_docks:
            print('Dock with ID {} closed'.format(dock_calling.id))
            del self.plot_docks[hash(dock_calling)]
        else:
            print('Unknown dock closed with ID {}'.format(dock_calling.id))

    def add_dock_plot(self, data=None, modifier=Modifier.mag_db):
        dock = PlotDock(data=data, modifier=modifier, closable=True)
        dock.setParent(self.dock_area)
        # Keep track of the dock in the plot lists (__hash__ is used to calculate the key)
        self.plot_docks[hash(dock)] = dock

        # Place docks:
        #   when this is the first dock -> place on the right
        #   when we already have a dock -> Place above (tabs)
        next_dock = next(iter(self.plot_docks.values()))
        self.dock_area.addDock(dock, 'right')  # Place dock to the right
        if len(self.plot_docks) > 0:
            self.dock_area.moveDock(dock, 'above', next_dock)

        dock.sigClosed.connect(self.dock_close_event)
        return dock

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
