import numpy as np
import pandas as pd
from OnePort import OnePort
from TwoPort import TwoPort
from CircElement import Zload, Resistor, Capacitor

# TODO: Store the circuit element as attribute
# TODO: Update the calc to multiply all previous ABCD matrices.

class Component(OnePort):
    # Give each instance a unique ID
    _ID = 0
    def __init__(self, component=None, id=None, Z0=50.0):
        super(Component, self).__init__(Z0=Z0)
        # Create an ID when needed
        if id is None:
            self.id = self._ID;
            self.__class__._ID += 1
        else:
            self.id = id

        # Dictionary of parameters
        self.parameters = dict()
        # Set the given component
        if component is None:
            self.component=TwoPort(Z0=self.Z0)
        else:
            self.component=component

        # Links to the components on the load and source side
        # The links are made from the source to the load.
        self.comp_src = None
        self.comp_load = None

    ##############################
    # Functions for creating an  #
    # iterator.                  #
    ##############################
    def __iter__(self):
        '''
        Returns the next component in the linked list. If there is no other component it raise an StopIteration.
        :return: The next item in the list
        '''
        here = self
        while here:
            yield here
            if here.comp_load is None:
                raise StopIteration
            here = here.comp_load

    def __reversed__(self):
        '''
        Returns the next component in the linked list. If there is no other component it raise an StopIteration.
        :return: The next item in the list
        '''
        here = self
        while here:
            yield here
            if here.comp_src is None:
                raise StopIteration
            here = here.comp_src

    ##############################
    # Functions for printing the #
    # complete circuit.          #
    ##############################
    def __str__(self):
        source = str(self.comp_load)

        # Add this component to the string
        comp_str = self.__repr__()
        # Add the parameters to the print out
        for key, value in self.parameters.items():
            comp_str += '\n\t{}: {}'.format(key, value)

        return '{}\n{}'.format(source, comp_str)

    def __repr__(self):
        return 'Component(component={}, id={})'.format(self.component, self.id)

    ##############################
    # Core s11 functionality     #
    ##############################
    def calc(self, freq):
        '''
        Returns the S11 at the given frequency. The frequency can be a list. First look in the cache. If it is not
        found it is calculated.
        :param freq:    frequency (can be list) in Hz
        :return:        List of S11, one for every frequency
        '''

        # If the frequency is not iterable (has getitem or iter) -> change it into a list
        if not hasattr(freq, '__getitem__'):
            freq = [freq]

        # Create list of frequencies not present yet
        freq_not_in_self = [freq_ for freq_ in freq if freq_ not in self.freq]
        if len(freq_not_in_self) == 0:
            # print('!! No new frequencies to calculate -> return self')
            return self

        # Are we a one-port (load)
        new_data = None
        if self.component.nports == 1:
            # print('!! We are the load -> stop calc')
            new_data = self.component.calc(freq_not_in_self)
        elif self.comp_load is not None:
            # print('-> Go into the load')
            vec_load = self.comp_load.calc(freq_not_in_self)
            # print('<- Returned')
        else:
            # No component on the load side
            # print('!! No load found, terminate with Z0')
            # We are no load -> terminate with the characteristic impedance
            vec_load = Zload(z=self.Z0)
            vec_load.calc(freq_not_in_self)

        # Multiply this component with the load impedance
        if new_data is None:
            new_data = self.component * vec_load

        # Store the data
        # Merge with the current S parameter data
        self._S = pd.concat([self._S, new_data._S])
        # Remove double frequencies -> Keep the original row
        self._S = self._S[~self._S.index.duplicated(keep='first')]

        return new_data

    ##############################
    # Linked list type operators #
    ##############################
    def invalidate(self):
        # Clear the local _S data (our local cache)
        self._S = self.default_matrix()

        # Invalidate everything towards the source side
        try:
            self.comp_src.invalidate()
        except AttributeError:
            pass

    def get_source(self):
        '''
        Get the source component in this linked list
        :return: the source component
        '''
        try:
            source = self.comp_src.get_source()
        except AttributeError:
            source = self

        return source

    def get_load(self):
        try:
            load = self.comp_low.get_load()
        except AttributeError:
            load = self

        return load

    def add_component(self, component=None, load_id=None):
        '''
        Adds a component to the source side of this component. When an ID id given check whether is matches this
        component if not -> pass on the load side. If ID==None than we simply add it. This function returns the
        reference to the newly created component.

        :param component:   Component to be added. If none: create a new component
        :param load_id:     ID of the component at the load side of the newly created component
        :return:            Reference to newly created component
        '''
        # Create component when not specified
        comp_new = Component(component=component)

        # Add component on the source side of this component when the id's match.
        if (load_id is None) or (self.id == load_id):
            prev_src = self.comp_src
            try:
                # Place the new component on the load side of the component at this source side.
                self.comp_src.comp_load = comp_new
            except AttributeError:
                pass

            # Add the source and load components to the new component
            comp_new.comp_src = self.comp_src
            comp_new.comp_load = self
            # Add the newly created component on the source of this one
            self.comp_src = comp_new
        else:
            comp_new = self.comp_load.add_component(component, load_id)

        return comp_new

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Add load
    load = Component(component=Zload(z=50.0))
    rpar = load.add_component(component=Resistor(R=50.0))
    rser = rpar.add_component(component=Resistor(R=10.0, series=True))
    rser.add_component(component=Capacitor(C=1e-9, series=True))

    print('\nPrint components from load to source:\n')
    src = load.get_source()
    print(src)

    # Calculate @ 2GHz
    src.calc(2e9)
    print(src.z11[None])

    # Iterate
    print('\nNow iterate, Print from source to load\n')
    head = load.get_source()
    for comp in head:
        print(comp.z11[1e9])

    print('\nCalc for a range of frequencies')
    freq = np.logspace(5, 9, 11)
    z = src.z11[freq]
    y = src.y11[freq]
    print(z)
    print(y)

    plt.loglog(freq, np.abs(z))
    plt.show()

