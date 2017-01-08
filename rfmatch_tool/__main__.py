import sys
import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore
#from PyQt5 import QtGui, QtWidgets

# Plotting libraty
import pyqtgraph as pg
from pyqtgraph.dockarea import *

from gui.main_window import Ui_MainWindow
from components.Circuit import Circuit

from enum import Enum
class Modifier(Enum):
    real = 1
    imag = 2
    mag_lin = 3
    mag_db = 4
    phase = 5

    def apply(self, data):
        if self.name == 'real':
            return np.real(data)
        elif self.name == 'imag':
            return np.imag(data)
        elif self.name == 'mag_lin':
            return np.abs(data)
        elif self.name == 'mag_db':
            return 10*np.log10(np.abs(data))
        elif self.name == 'phase':
            return np.angle(data, deg=True)
        else:
            raise AttributeError('Unknown modifier {}'.format(self.name))

    def unit(self):
        if self.name == 'real':
            return ''
        elif self.name == 'imag':
            return ''
        elif self.name == 'mag_lin':
            return ''
        elif self.name == 'mag_db':
            return 'dB'
        elif self.name == 'phase':
            return 'deg'
        else:
            raise AttributeError('Unknown modifier {}'.format(self.name))


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
        self.default_dock_setup()

        # Update all gui elements with the datamodel
        self.update_gui_data()

        # Connect buttons
        self.pbPlot.clicked.connect(self.add_plot)
        self.pbAddPoints.clicked.connect(self.add_points)

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

        freq = np.linspace(x_start, x_stop, x_num)
        self.circuit.circuit.calc(freq)

        self.dock_update()

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

    #######################
    # Plot dock functions #
    #######################
    def dock_update(self):
        for key, dock in self.plot_docks.items():
            dock.update()

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

class PlotDock(Dock):
    _ID = 0

    def __init__(self, *args, **kwargs):
        # Get the data and modifier from the arglist
        data = kwargs.pop('data', None)
        self.data = data
        kwargs['name'] = self.data_name
        modifier = kwargs.pop('modifier', Modifier.mag_db)
        super(PlotDock, self).__init__(*args, **kwargs)

        # Keep track of the ID (give each dock an unique ID)
        self.__id = self._ID
        self.__class__._ID += 1

        # Setup the Plot + data
        self._modifier = modifier
        self.index_range = None                         # What data to plot. None=all data

        # Plot
        self.plot_widget = pg.PlotWidget(title=self.data_name)
        self.addWidget(self.plot_widget)
        self.curve = None

        self.update()

        # Setup the dock (title)
        self.setTitle(self.data_name)
        self.autoOrient = False
        self.setOrientation('horizontal', force=True)

        self.setup_crosshair()

    def update(self):
        #self.plot_widget.close()
        self.plot_widget.plotItem.clear()
        self.update_cache()
        self.plot()
        self.set_marker()

    def plot(self):
        self.curve = self.plot_widget.plot(self.x, self.y)

        # Apply defaults
        self.defaults()

        # Annotate
        self.annotate()

    def annotate(self):
        y_axis = '{}({})'.format(self._modifier.name, self.data_name)
        y_unit = self._modifier.unit()
        self.plot_widget.setLabel('left', y_axis, units=y_unit)
        self.plot_widget.setLabel('bottom', "Frequency", units='Hz')

    def defaults(self):
        self.plot_widget.setLogMode(x=False, y=False)

    #############################
    # Data cache functions      #
    # The data is cached to     #
    # speed up all data access  #
    # functions                 #
    #############################
    def update_cache(self):
        self._x_cache = self.data[self.index_range].index.values
        self._y_cache = self._modifier.apply(self.data[self.index_range].values)

        # sort
        idx = np.argsort(self._x_cache)
        self._x_cache = self._x_cache[idx]
        self._y_cache = self._y_cache[idx]

    @property
    def x(self):
        return self._x_cache

    @property
    def y(self):
        return self._y_cache

    ########################
    # Cross hair functions #
    ########################
    def setup_crosshair(self):
        # cross hair
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        self.plot_widget.addItem(self.hLine, ignoreBounds=True)

        # marker
        self.set_marker()

        self.vb = self.plot_widget.plotItem.vb

        proxy = pg.SignalProxy(self.plot_widget.scene().sigMouseMoved, rateLimit=2, slot=self.mouseMoved)
        self.plot_widget.scene().sigMouseMoved.connect(self.mouseMoved)

    def mouseMoved(self, evt):
        pos = evt
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mousePoint = self.vb.mapSceneToView(pos)
            index = mousePoint.x()

            if self.logx:
                index = 10**index

            index_snap = self.find_nearest(self.x, index)
            self.update_marker(index_snap)

            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())

    def set_marker(self):
        ## Set up a marker
        self.curvePoint = pg.CurvePoint(self.curve)
        self.plot_widget.addItem(self.curvePoint)
        self.marker_label = pg.TextItem("test", anchor=(0.5, -1.0))
        self.marker_label.setParentItem(self.curvePoint)
        self.marker_arrow = pg.ArrowItem(angle=90)
        self.marker_arrow.setParentItem(self.curvePoint)

    def reset_marker(self):
        self.curvePoint = pg.CurvePoint(self.curve)

    def update_marker(self, index):
        xnum = len(self.x) #data[None].index.values)
        xindex = np.argmin(np.abs(self.x - index))
        xdata = index
        ydata_mod = self.y[xindex]

        self.curvePoint.setPos((xindex+1)/xnum)
        self.marker_label.setText('[%0.1f, %0.1f]' % (xdata, ydata_mod))

    def find_nearest(sefl, array, value):
        """
        Very slow implementation to find a nearest value in an array (used to snap the cross hair)
        :param array:
        :param value:
        :return:
        """
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    #########################
    # High level properties #
    #########################
    @property
    def data_name(self):
        """
        Return the name of the data (s11, z21, etc). Return none when unknown
        :return:
        """
        try:
            title = self.data.varname
        except:
            title = 'None'

        return title

    @property
    def modifier(self):
        return self.modifier.name

    @modifier.setter
    def modifier(self, value):
        if value in Modifier.__members__.keys():
            self._modifier = Modifier[value]
        elif value in Modifier:
            self._modifier = value
        else:
            raise AttributeError('Unknown modifier {}'.format(value))

    @property
    def logx(self):
        try:
            state = self.plot_widget.plotItem.ctrl.logXCheck.isChecked()
        except:
            state = False

        return state

    @property
    def logy(self):
        try:
            state = self.plot_widget.plotItem.ctrl.logYCheck.isChecked()
        except:
            state = False

        return state

    ################################
    # Hashable + compare instances #
    ################################
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
