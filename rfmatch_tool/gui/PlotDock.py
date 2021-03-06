# Plotting libraty
import pyqtgraph as pg
from pyqtgraph.dockarea import *
from pyqtgraph.graphicsItems.ROI import CrosshairROI

import numpy as np

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

        # The max allowed value
        self._max_y_val = 1e10 #np.finfo(np.float64).max

        # Setup the Plot + data
        self._modifier = modifier
        self.index_range = None                         # What data to plot. None=all data

        # Setup the dock (title)
        self.setTitle(self.data_name)
        self.autoOrient = False
        self.setOrientation('horizontal', force=True)

        # Add Plot area and empty curve
        self.plot_widget = pg.PlotWidget(title=self.data_name)
        self.addWidget(self.plot_widget)
        self.vb = self.plot_widget.plotItem.vb
        self.curve = None               # Store the curve

        self.update()

        self.setup_crosshair()

    def update(self, data=None):
        # Set the new data (when given)
        if data is not None:
            self.data = data

        # Update the plot itself
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

        # Replace infinite
        idx_posinf = (self._y_cache ==  np.inf)
        idx_neginf = (self._y_cache == -np.inf)
        self._y_cache[idx_posinf] = self._max_y_val
        self._y_cache[idx_neginf] = -self._max_y_val

    @property
    def x(self):
        return self._x_cache

    @property
    def y(self):
        return self._y_cache

    ########################
    # Cross hair functions #
    ########################
    def setup_crosshair(self, kind='cross'):
        """
        Setup of the cursor. Can select the type of cursor and snapping you want
        :param kind: Cursor type. Options:
                        'cross': both vertical and horizontal, snap to nearest point
                        'horiz': horizontal, snap on y values
                        'vert': vertical snap on  values
        :return:
        """
        if kind not in ['cross', 'horiz', 'vert']:
            raise ValueError('Unknown cursor type kind={}'.format(kind))

        # By default the cursors are off
        self.hLine = None
        self.vLine = None
        if kind in ['cross', 'horiz']:
            self.hLine = pg.InfiniteLine(angle=0, movable=False)
            self.plot_widget.addItem(self.hLine, ignoreBounds=True)
        if kind in ['cross', 'vert']:
            self.vLine = pg.InfiniteLine(angle=90, movable=False)
            self.plot_widget.addItem(self.vLine, ignoreBounds=True)

        # marker
        self.set_marker()

        proxy = pg.SignalProxy(self.plot_widget.scene().sigMouseMoved, rateLimit=2, slot=self.mouseMoved)
        self.plot_widget.scene().sigMouseMoved.connect(self.mouseMoved)

    def mouseMoved(self, evt):
        pos = evt
        if self.plot_widget.sceneBoundingRect().contains(pos):
            # Convert the screen coord to the coord inside the plotwidget
            mousePoint = self.vb.mapSceneToView(pos)

            # Convert the widget-space point from log when needed
            xdata = mousePoint.x()
            if self.logx:
                xdata = 10**xdata

            # Snap the point to the nearest curve point
            xdata_snap = self.find_nearest(self.x, xdata)
            self.update_marker(xdata_snap)

            # Update the crosshair
            snap_x=snap_y=0
            if self.vLine is not None:
                self.vLine.setPos(mousePoint.x())
                snap_x = True
            if self.hLine is not None:
                self.hLine.setPos(mousePoint.y())
                snap_y = True

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

    def update_marker(self, xdata):
        xnum = len(self.x) #data[None].index.values)
        xindex = np.argmin(np.abs(self.x - xdata))
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
