import numpy as np
import pandas as pd
from math import pi
import scipy.constants as Const

#import matplotlib.pyplot as plt

from .TwoPort import TwoPort
from .OnePort import OnePort

class CircElement(TwoPort):
    # Store the component type
    _elem_type = 'CircElement'

    """Class derived from TwoPort, used as parent class for circuit elements.
        Implements methods to convert the series/parallel impedance to an ABCD matrix."""
    def __init__(self, freq=1e9, series=False, Z0=50.):
        """Constructor of class CircElement"""
        super(CircElement, self).__init__(Z0=Z0)

        # Dictionary of parameters
        self.parameters = {'series': series}

    def calc_z(self, freq):
        return 0.0

    def calc_abcd(self, freq):
        """
        Calculate the impedance given frequency freq
        """
        z = self.Calc_z(freq)

        if self.parameters['series']:
            abcd = np.array([1.0, z, 0.0, 1.0], dtype=np.complex_)
        else:  # Parallel connection
            abcd = np.array([1.0, 0.0, 1 / z, 1.0], dtype=np.complex_)

        return pd.DataFrame(data = [abcd], columns=['A', 'B', 'C', 'D'], index=[freq])

    def calc(self, freq):
        try:
            df_list = [self.Calc_abcd(freq_) for freq_ in freq]
            df_abcd = pd.concat(df_list)
        except TypeError:
            # Not itteratable
            df_abcd = self.Calc_abcd(freq)

        # Convert ABCD to S
        df_s = df_abcd.groupby(df_abcd.index).apply(self._ABCD_to_S)
        # Merge with the current S parameter data
        self._S = pd.concat([self._S, df_s])
        # Remove double frequencies -> Keep the original row
        self._S = self._S[~self._S.index.duplicated(keep='first')]

        return self

    @property
    def name(self):
        _name = self.__class__.__name__

        if self.parameters['series']:
            _name += '_series'
        else:
            _name += '_shunt'

        return _name

class Inductor(CircElement):
    def __init__(self, L=1e-9, Q=None, srf=None, *kwargs):
        super(Inductor, self).__init__(*kwargs)

        # Store parameters
        self.parameters['L'] = L            # Inductance
        self.parameters['Q'] = Q            # Quality factor
        self.parameters['srf'] = srf        # Self resonance frequency

    # str and repr functions for easy printing
    def __repr__(self):
        return 'Inductor(L={}, Q={}, srf={}, series={}, Z0={})'.format(self.parameters['L'], self.parameters['Q'], self.parameters['srf'], self.parameters['series'], self.Z0)

    def __str__(self):
        # Call the super to show the complete linked list
        str_super = super(Inductor, self).__str__()
        return '{}{}'.format(str_super, self.__repr__())

    # Overload the calculator
    def calc_z(self, freq=1e9):
        """Calculate the effective impedance at the set frequency"""
        w = 2*pi * freq
        # Calculate the series resistance
        if self.parameters['Q'] is None:
            rs = 0.
        else:
            rs = w*self.parameters['L'] / self.parameters['Q']

        # Calculate the parallel capacitance
        if self.parameters['srf'] is None:
            y_cpar = 0
        else:
            cpar = 1 / ((2*pi*self.parameters['srf'])**2 * self.parameters['L'])
            y_cpar = 1j*w * cpar

        zl = 1j*w*self.parameters['L']        # Only the inductor
        yl = 1/(zl + rs)        # Inductor in series with DC resistance
        z = 1/(yl+y_cpar)       # Complete inductor with res and src cap

        return z

class Capacitor(CircElement):
    def __init__(self, C=1e-9, Q=None, srf=None, series=False, *kwargs):
        super(Capacitor, self).__init__(series=False, *kwargs)
        # Store local variables
        self.parameters['C'] = C          # Capacitance
        self.parameters['Q'] = Q          # Quality factor
        self.parameters['srf'] = srf      # Self resonance frequency

    # str and repr functions for easy printing
    def __repr__(self):
        return 'Capacitor(C={}, Q={}, srf={}, series={}, Z0={})'.format(self.parameters['C'], self.parameters['Q'], self.parameters['srf'], self.parameters['series'], self.Z0)

    def __str__(self):
        # Call the super to show the complete linked list
        str_super = super(Capacitor, self).__str__()
        return '{}{}'.format(str_super, self.__repr__())

    # Overload the calculator
    def calc_z(self, freq):
        """Calculate the effective impedance at the set frequency"""
        w = 2*pi*freq
        # Calculate the series resistance
        if self.parameters['Q'] is None:
            rs = 0.
        else:
            rs = w*self.parameters['C'] / self.parameters['Q']

        # Calculate the parallel capacitance
        if self.parameters['srf'] is None:
            z_lser = 0
        else:
            lser = 1 / ((2*pi*self.parameters['srf'])**2 * self.parameters['C'])
            z_lser = 1j*w * lser

        yc = 1j*w*self.parameters['C']        # Only the capacitor
        z = z_lser + rs + 1/yc  # Complete capacitor with res and series inductor

        return z

class Resistor(CircElement):
    def __init__(self, R=50., series=False, *kwargs):
        super(Resistor, self).__init__(series=series, *kwargs)
        # Store local variables
        self.parameters['R'] = R          # Resistance

    # str and repr functions for easy printing
    def __repr__(self):
        return 'Resistor(R={}, series={}, Z0={})'.format(self.parameters['R'], self.parameters['series'], self.Z0)

    def __str__(self):
        # Call the super to show the complete linked list
        str_super = super(Resistor, self).__str__()
        return '{}{}'.format(str_super, self.__repr__())

    # Overload the calculator
    def calc_z(self, freq=None):
        """Calculate the effective impedance at the set frequency"""
        return self.parameters['R']

class Zload(OnePort):
    """Class derived from OnePort, used as parent class for circuit elements.
        Implements methods to convert the series/parallel impedance to an ABCD matrix."""
    def __init__(self, z = 50.0, Z0=50.):
        """Constructor of class CircElement"""
        super(Zload, self).__init__(Z0=Z0)
        self.parameters = {'Z': z}

    def calc_z(self, freq):
        return self.parameters['Z']

    def calc_z11(self, freq):
        """
        Calculate the impedance given frequency freq
        """
        z11 = self.Calc_z(freq)

        return pd.DataFrame(data = [z11], columns=['z11'], index=[freq])

    def calc(self, freq):
        try:
            df_list = [self.Calc_z11(freq_) for freq_ in freq]
            df_z = pd.concat(df_list)
        except TypeError:
            # Not itteratable
            df_z = self.Calc_z(freq)

        # Convert Z to S
        df_s = df_z.groupby(df_z.index).apply(self.Z_to_S)
        # Merge with the current S parameter data
        self._S = pd.concat([self._S, df_s])
        # Remove double frequencies -> Keep the original row
        self._S = self._S[~self._S.index.duplicated(keep='first')]

        return self

    ##############################
    # Functions for printing     #
    ##############################
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'Zload(z={}, Z0={})'.format(self.parameters['Z'], self.Z0)

    @property
    def name(self):
        return 'zload'


class TLine(CircElement):
    def __init__(self, l=100e-6, alpha=0, eps_r=1.0, mu_r=1.0, Z0=50.0, *kwargs):
        super(TLine, self).__init__(*kwargs)
        self.parameters['L'] = l
        self.alpha  = alpha
        self.eps_r  = eps_r
        self.mu_r   = mu_r
        self.Z0     = Z0

    def calc_abcd(self, freq):
        """
        Calculate the impedance given frequency freq
        """
        # TODO: Implement the series / parallel connection
        # Calculate wavelength
        wavelength = self.get_wavelength(freq)
        # Calculate beta
        beta = 2*np.pi / wavelength
        # Combine to gamma
        gamma = complex(self.alpha, beta)

        # Calculate ABCD matrix
        A = np.cosh(gamma * self.parameters['L'])
        B = np.sinh(gamma * self.parameters['L']) * self.Z0
        C = np.sinh(gamma * self.parameters['L']) / self.Z0
        D = np.cosh(gamma * self.parameters['L'])

        return pd.DataFrame(data = [[A, B, C, D]], columns=['A', 'B', 'C', 'D'], index=[freq])

    def get_wavelength(self, freq):
        # Calculate speed of light in medium
        v = 1/np.sqrt(Const.mu_0 * self.mu_r * Const.epsilon_0 * self.eps_r)
        # Calculate and return the wavelength
        return v / freq

    @property
    def wavelength(self):
        return self.get_wavelength(self.freq)

class DataPort(TwoPort):
    def __init__(self, *kwargs):
        super(DataPort, self).__init__(*kwargs)
        self.empty()

    def empty(self):
        '''
        Clear the s2p port data.
        :return: Nothing
        '''
        self.df_port_data = pd.DataFrame(data=None, columns=['freq', 's11', 's12', 's21', 's22'])

    def load_s2p(self, filename):
        '''
        Load 2-port s parameters from a touchstone file (.s2p). It returns a pandas dataframe containing the data.
        :param filename:    Filename of the touchstone file
        :return:            Pandas dataframe containing the columns: freq, s11, s21, s12, s22
        '''
        # Open File
        fh = open(filename)

        # Create an empty dataframe
        self.empty()

        #
        # Parse the option line
        #
        # Search for the option line
        line = fh.readline()
        while line[0] != '#':
            line = fh.readline()

        # Line found parse options (start after # symbol) -> convert to lower case -> strip options (remove white spaces)
        # options:
        #       0:  Frequncy Units: Hz, kHz, MHz, GHz (default)
        #       1:  Parameter: Z, Y, H, G, S (default)
        #       2:  format: dB (db/angle deg) ; RI (real/imag) ; MA (mag/angle deg) (default
        #       3:  R
        #       4:  n: reference resistance (default=50)
        options = line[1:].lower().strip().split()

        # Frequency units
        if options[0] == 'hz':
            freq_scaler = 1.0
        elif options[0] == 'khz':
            freq_scaler = 1e3
        elif options[0] == 'mhz':
            freq_scaler = 1e6
        else:  # GHz is the default
            freq_scaler = 1e9

        # Z0
        self.Z0 = float(options[4])

        # Loop through the file
        line = fh.readline()
        while line:
            # Do not read comments
            if ((line[0] == '#') or (line[0] == '!')):
                continue

            # Convert read string to float
            data_line = map(float, line.split(','))
            # Save the frequency information
            self.freq += [data_line[0] * freq_scaler]
            # Create empty value list (only used for loadpull)
            self.Loadpull_values += [0.0]

            # Convert read data to complex(real, imag)
            if options[2] == 'db':
                val_complex = 10 ** (np.array(data_line[1::2]) / 10.) * np.exp(1j * np.array(data_line[2::2]))
            elif options[2] == 'ma':
                val_complex = np.array(data_line[1::2]) * np.exp(1j * np.pi / 180. * np.array(data_line[2::2]))
            else:  # options[3] == 'ri'
                val_complex = np.array(data_line[1::2]) + 1j * np.array(data_line[2::2])

            # Save data as NPORT
            self.port_data += [TwoPort.TwoPort(Z0=self.Z0)]
            if options[1] == 'z':
                self.port_data[-1].Z = np.matrix(np.reshape(val_complex, (2, 2)).transpose())
            elif options[1] == 'y':
                self.port_data[-1].Y = np.matrix(np.reshape(val_complex, (2, 2)).transpose())
            else:  # S-Parameter
                self.port_data[-1].S = np.matrix(np.reshape(val_complex, (2, 2)).transpose())

            # Read next line
            line = fh.readline()

        return self

    def calc(self, freq=1e9):
        # Calculate wavelength
        wavelength = self.wavelength(freq)
        # Calculate beta
        self.beta = 2*np.pi / wavelength
        # Combine to gamma
        gamma = complex(self.alpha, self.beta)

        # Calculate ABCD matrix
        A = np.cosh(gamma * self.parameters['L'])
        B = np.sinh(gamma * self.parameters['L']) * self.Z0
        C = np.sinh(gamma * self.parameters['L']) / self.Z0
        D = np.cosh(gamma * self.parameters['L'])

        port = TwoPort(Z0 = self.Z0)
        port.ABCD = np.array([ [A, B], [C, D] ])
        return port

if __name__ == '__main__':
    print('Create an inductor and calculate at 1 frequency')
    ind1 = Inductor()
    freq = 1e9
    ind1.calc(freq)
    print(ind1)

    print('\nCreate an inductor and calculate at 2 frequencies')
    ind2 = Inductor()
    ind2.series = True
    freq = np.array([1e9, 2e9])
    ind2.calc(freq)
    print(ind2)

    print('\nZload test')
    load = Zload(z=50.0)
    load.calc([1e9, 2e9, 3e9, 4e9])
    print(load.z11[None])

    print('\nMultiply the inductor and the load (place in series)')
    ind_load = ind2 * load
    print(ind_load.z11[None])

    print('\nPlace a 50 Ohm Resistor parallel to another 50Ohm load')
    rpar = Resistor(R=50.0)
    rpar.series = False
    rpar_load = rpar * load
    print(rpar_load.z11[None])

    print('\nTest TLine (Quarter lambda @ 3GHz)')
    tline = TLine(eps_r = 1.0, Z0=100, l=0.1/4)
    tline_load = tline * load
    print('Wavelengths: {}'.format(tline.wavelength))
    print(tline_load.z11[None])

    print('\nTest inductor with srf=5e9 in series with Rload')
    lres = Inductor(L=1e-9, srf=5e9)
    freq = np.logspace(7, 11, 101)
    lres.calc(freq)
    lres_load = lres * load
    # plt.semilogx(lres_load.z11[None])
    # plt.show()