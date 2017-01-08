import numpy as np
import pandas as pd

from .TwoPort import TwoPort
from .CircElement import Zload, Resistor, Capacitor


class Component(TwoPort):
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
    # component.                 #
    ##############################
    def __str__(self):
        source = str(self.comp_load)

        # Add this component to the string
        comp_str = self.__repr__()
        # Add the parameters to the print out (when we have parameters)
        try:
            for key, value in self.parameters.items():
                comp_str += '\n\t{}: {}'.format(key, value)
        except:
            pass

        return comp_str

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
        :return:        Depending on the component contained, it either returns:
                        - TwoPort of the combination of this component up to the load (series)
                        - None when a OnePort is stored (this is the load)
        '''

        # If the frequency is not iterable (has getitem or iter) -> change it into a list
        if not hasattr(freq, '__getitem__'):
            freq = [freq]

        # Create list of frequencies not present yet
        freq_not_in_self = [freq_ for freq_ in freq if freq_ not in self.freq]
        if len(freq_not_in_self) == 0:
            # print('!! No new frequencies to calculate -> return self')
            return self

        # Calculate the s parameters of the locally stored component.
        self.component.calc(freq_not_in_self)

        new_data = None
        # Are we a one-port (load)
        if self.component.nports == 1:
            # print('!! We are the load -> stop calc')
            raise NotImplementedError('Currently it is not possible to add a 1-Port to the Component list')
        elif self.comp_load is not None:
            # print('-> Go into the load')
            comp_load = self.comp_load.calc(freq_not_in_self)
            if comp_load is None:
                # Load is not a TwoPort -> new_data is the locally stored component
                new_data = self.component
            else:
                # The new data is the multiplication of the load matrix and the locally stored component
                new_data = self.component * comp_load
            # print('<- Returned')
        else:
            # No component on the load side -> return self
            new_data = self.component

        # Store the data in this component (the multiplication of all previous components)
        # Merge with the current S parameter data
        self._S = pd.concat([self._S, new_data._S])
        # Remove double frequencies -> Keep the original row
        self._S = self._S[~self._S.index.duplicated(keep='first')]

        return new_data

    ##############################
    # Parameter functions        #
    ##############################
    @property
    def circuit_parameters(self):
        if self.component is None:
            return ''
        else:
            params = ['{}: {}'.format(key, value) for key,value in self.component.parameters.items()]
            return '\n'.join(params)

    @property
    def name(self):
        if self.component is None:
            return ''
        else:
            return self.component.name

    ##############################
    # Linked list type operators #
    ##############################
    def __len__(self):
        """
        Return the number of elements
        :return: (int) Number of elements
        """
        src = self.get_source()
        return src._len_count()

    def _len_count(self):
        """
        Count the number of elements on the load side
        :return: (int) number of elements on the load side (including this component)
        """
        if self.comp_load is not None:
            return self.comp_load._len_count() + 1
        else:
            return 1

    def get_index(self, index):
        """
        Return the item at the given index. If index=0 return self (required index)
        :param index:
        :return:
        """
        if index == 0:
            return self
        if self.comp_load is not None:
            return self.comp_load.get_index(index-1)
        else:
            raise IndexError('Index not found, max index is: {}'.format(len(self)))

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
        # Test to see whether the component to be added is a TwoPort
        if component.nports != 2:
            raise NotImplementedError('Currently it is only possible to add a 2-Port to the Component list')

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
    load = Zload(z=50.0)
    rpar = Component(component=Resistor(R=50.0))
    rser = rpar.add_component(component=Resistor(R=10.0, series=True))
    rser.add_component(component=Capacitor(C=1e-9, series=True))

    print('\nPrint components from load to source:\n')
    src = rpar.get_source()
    print(src)

    # Calculate @ 2GHz
    src.calc(2e9)
    print(src.z11[None])

    # Iterate
    print('\nNow iterate, Print from source to load\n')
    head = rpar.get_source()
    for comp in head:
        print(comp.z11[1e9])

    print('\nCalc for a range of frequencies')
    freq = np.logspace(5, 9, 11)
    src.calc(freq)
    #load.calc(freq)
    port_in = src * load
    z = port_in.z11[freq]
    y = port_in.y11[freq]
    print(z)
    print(y)

#    plt.loglog(freq, np.abs(z))
#    plt.show()

