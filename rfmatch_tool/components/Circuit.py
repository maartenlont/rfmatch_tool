import numpy as np
from .OnePort import OnePort
from .CircElement import Zload, Resistor, Capacitor
from .Component import Component
#import matplotlib.pyplot as mpl

from PyQt5.QtCore import *
from PyQt5.QtGui import *

class Circuit(QAbstractListModel):
    def __init__(self):
        super(Circuit, self).__init__()

        # Setup an empty circuit, load and source
        self.source = OnePort()
        self.load = OnePort()
        self.circuit = Component()

        # Initialize the circuits
        self.circuit_init()
        self.load = self.circuit.get_load()

    def circuit_init(self):
        """
        Create a simple circuit for easy testing
        :return:
        """
        # Set the load and source impedances
        self.load = Zload(z=50.0)
        self.source = Zload(z=50.0)

        # Create a simple circuit for easy testing
        # Load -> Shunt R -> Series R -> Series C
        self.circuit = Component(component=Zload(z=50.0))
        rshunt = self.circuit.add_component(component=Resistor(R=50.0))
        rser = rshunt.add_component(component=Resistor(R=10.0, series=True))
        rser.add_component(component=Capacitor(C=1e-9, series=True))

        # Set circuit to point to the source side
        self.circuit = self.circuit.get_source()

    @property
    def parameter_list(self):
        try:
            par_list = self.circuit.parameter_list
        except:
            par_list = list()

        par_list += ['zin']
        par_list += ['gamma']

        return par_list

    ################################
    # QAbstractListModel interface #
    ################################
    def get_item_at(self, index):
        # Get the required row number
        row = index.row()

        # Get the component at "row"
        if row == 0:
            # If row = 0 -> Return the source
            component_row = self.source
        elif row > len(self.circuit):
            raise IndexError('Row index {} out of range {}'.format(row, len(self.circuit)))
        else:
            component_row = self.circuit.get_index(row - 1)

        return component_row

    def remove_component(self, index):
        self.beginRemoveRows(index, index.row(), index.row())

        elem = self.get_item_at(index)

        # Remove the element and set the new source when needed
        self.circuit = elem.remove()

        self.endRemoveRows()

        return self.circuit

    def rowCount(self, parent=QModelIndex()):
        """
        Return the total # of elements including the source

        :param parent:
        :return:
        """
        # print("Circuit->rowCount()")
        try:
            circuit_len = len(self.circuit.get_source())
        except:
            circuit_len = 0

        # print("\t# of components in circuit: {}".format(circuit_len))

        return circuit_len + 1

    def columnCount(self, parent=QModelIndex()):
        # print("Circuit->columnCount()")
        return self.rowCount(parent)

    def data(self, index, role=Qt.DisplayRole):
        # print('Circuit->data()')
        # Catch when the index is not valid
        if not index.isValid() or not 0 <= index.row() < self.rowCount():
            return QVariant()

        row = index.row()
        component_row = self.get_item_at(index)

        # Print text (component type)
        if role == Qt.DisplayRole:
            try:
                id = component_row.name #str(component_row.id)
            except:
                if row == 0:
                    id = 'Source'
                else:
                    raise IndexError('Incorrect row index {}'.format(row))
            return id

        # As tooltop show all the circuit parameters (resistance, inductance, etc)
        if role == Qt.ToolTipRole:
            return component_row.circuit_parameters

        # Return the icon belonging to the component name
        if role == Qt.DecorationRole:
            filename = component_row.name.lower()
            comp_image = QImage('gui/images/{}.png'.format(filename))
            if row == 0:
                # Mirror the source
                comp_image = comp_image.mirrored(True, False)

            return QIcon(QPixmap.fromImage(comp_image))
#            return QIcon('gui/images/{}.png'.format(filename))

        return QVariant()

# Helper functions

if __name__ == "__main__":
    pass