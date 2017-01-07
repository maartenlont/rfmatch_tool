import numpy as np
import pandas as pd
import itertools
from functools import partial
import random
import copy
import unittest

def _to_matrix(group):
    """
    Convert the dataframe / series or array to 1 2D array. When the group is a DataFrame a DataFrame is returned with
    the same index.
    :param group:
    :return:
    """
    # Get the values of the sparameters
    flat_matrix = group.values
    # Reshape to a square 2-dimensional array and cast to matrix
    nports = int(np.sqrt(np.max(np.shape(flat_matrix))))
    matrix = np.reshape(flat_matrix, (nports, nports))
    # Clean matrix only (worked before)
    #return matrix
    if isinstance(group, pd.DataFrame):
        return pd.DataFrame({'matrix': [matrix]}, index=group.index)
    else:
        return matrix

def _to_flat_matrix(data):
    """
    Convert the dataframe / series or array to a 1D array. When the group is a DataFrame or Series the index is
    returned as well.
    :param group:
    :return: (index, data_1d)
    """
    if isinstance(data, pd.DataFrame):
        idx = data.index
        data_flat = data.values.reshape((1, -1))
    elif isinstance(data, pd.Series):
        idx = [data.name]
        data_flat = data.values.reshape((1, -1))
    else:
        idx = [0]
        try:
            data_flat = data.reshape((1, -1))
        except AttributeError:
            data_flat = np.array([data])

    return (idx, data_flat[0])

# S parameter iterator
class DataIterator:
    def __init__(self, data):
        self.index = 0
        self.data = data

    def __next__(self):
        try:
            value = self.data.iloc[self.index]
        except IndexError:
            raise StopIteration()

        self.index += 1
        return value

    def __iter__(self):
        return self

# Descriptor used to access s parameter
class DataDescriptor(object):
    def __init__(self, owner, varname):
        self.owner = owner
        self.varname = varname

    # def __get__(self, instance, owner):
    #     print('DataDescriptor[{}].__get__'.format(self.varname))
    #     return self.get_data(None)

    # def __set__(self, instance, value):
    #     print('DataDescriptor[{}].__set__'.format(self.varname))

    # Be able to use [...] to get the parameters from a frequency
    def __getitem__(self, freq):
        return self.get_data(freq)

    def __setitem__(self, freq, value):
        try:
            self.owner.set_parameter(freq, value, par_type=self.varname[0].upper(), par_name=self.varname)
        except:
            raise AttributeError('Unknown parameter {}'.format(self.varname[0]))

    # Iterator functions
    def __iter__(self):
        data = self.get_data(None)
        return DataIterator(data)

    def get_data(self, freq=None):
        try:
            data = self.owner.get_parameter(freq, par_type=self.varname[0].upper())
        except:
            raise AttributeError('Unknown parameter {}'.format(self.varname[0]))
        return data[self.varname]

# Descriptor used to access s parameter (returns the full matrix)
class DataMatrixDescriptor():
    def __init__(self, owner, varname):
        self.owner = owner
        self.varname = varname

    # Be able to use [...] to get the parameters from a frequency
    def __getitem__(self, freq):
        # Get s parameters and convert it to a matrix
        port_names = self.owner.get_col_names(self.varname)
        df_spars = self.get_data(freq)
        if isinstance(df_spars, pd.Series):
            df_matrix = _to_matrix(df_spars[port_names])
        elif isinstance(df_spars, pd.DataFrame):
            df_matrix = df_spars.groupby(df_spars.index)[port_names].apply(_to_matrix)
            df_matrix = df_spars.merge(df_matrix, left_index=True, right_index=True)
        else:
            raise ValueError('The {} parameter data is not a proper pandas object!'.format(self.varname))
        return df_matrix

    def __setitem__(self, freq, value):
        df_spars = self.set_data(freq, value)

    def __set__(self, instance, value):
        for freq in value.index.values:
            self.set_data(freq, value.loc[freq])
#        print('!!DataMatrixDescriptor.__set__')

    # Iterator functions
    def __iter__(self):
        data = self.get_data()
        port_names = self.owner.get_col_names(self.varname)
        df_matrix = data.groupby(data.index).apply(_to_matrix)
        return DataIterator(df_matrix)

    def __str__(self):
        str_ret = 'DataMatrixDescriptor({})\n'.format(self.varname)
        str_ret += str(self.get_data())
        return str_ret

    def get_data(self, freq=None):
        try:
#            data = getattr(self.owner, 'get_{}'.format(self.varname.lower()))(freq)
            data = self.owner.get_parameter(freq, par_type=self.varname.upper())
        except:
            raise AttributeError('Unknown parameter {}'.format(self.varname))
        return data

    def set_data(self, freq, value):
        try:
#            data = getattr(self.owner, 'set_{}'.format(self.varname.lower()))(freq, value)
            data = self.owner.set_parameter(freq, value, par_type=self.varname.upper())
        except:
            raise AttributeError('Unknown parameter {}'.format(self.varname))
        return data


class NPort(object):
    # Constructors and matrix generation
    def __init__(self, freq=None, nports=2, Z0=50., S=None):
        self.__nports = nports
        self.Z0=Z0
        if S is None:
            self._S = self.default_matrix(freq)
        else:
            self._S = copy.deepcopy(S)

        # list all parameters
        self.parameter_list = list()

        # Create list of ports
        ports = range(1, self.nports+1)
        ports_comb = itertools.product(ports, ports)

        # Create functions that return the s,y,z parameters
        for par_type_ , port_ in itertools.product(['s', 'y', 'z'], ports_comb):
            name = self.get_col_name(port_, par_type=par_type_)
            self.__dict__[name] = DataDescriptor(self, name)
            # Also add to parameter list
            self.parameter_list += [name]

        # Get S,Y,Z parameter matrix
        self.__dict__['s'] = DataMatrixDescriptor(self, 's')
        self.__dict__['y'] = DataMatrixDescriptor(self, 'y')
        self.__dict__['z'] = DataMatrixDescriptor(self, 'z')

    # Overwrite the getattribute such that also instance attributes get called (_get_)
    # def __getattribute__(self, attr):
    #     obj = object.__getattribute__(self, attr)
    #     if hasattr(obj, '__get__'):
    #         return obj.__get__(self, type(self))
    #     return obj

    #############################
    # Calculate a new frequency #
    #############################
    def calc(self, freq):
        # Simply create a default matrix for each frequency point
        new_matrices = self.default_matrix(freq)

        # Merge with the current S parameter data
        self._S = pd.concat([self._S, new_matrices])
        # Remove double frequencies -> Keep the original row
        self._S = self._S[~self._S.index.duplicated(keep='first')]

        return self

    def generate_freq(self, freq_new):
        """
        Generate S parameter data for the given frequency list. (Only generate when the frequencies do not exist yet)
        :param freq_new: List of frequencies to generate
        :return: self
        """
        # Get a list of frequencies in one but not the other -> try to calculate the missing freqs
        freq_self = self.freq

        try:
            freq_not_in_self = [freq for freq in freq_new if freq not in freq_self]
        except TypeError: # freq_new is not a list
            if freq_new in freq_self:
                return self
            else:
                freq_not_in_self = [freq_new]

        self.calc(freq_not_in_self)

        return self

    def sync_freq(self, other):
        """
        Synchronize the frequencies in both NPorts. If the frequency does not exist in one of the two then calc is
        called on the missing frequencies.
        :param other: Other NPort to sync frequencies with
        :return: self
        """
        try:
            # Get a list of frequencies in one but not the other -> try to calculate the missing freqs
            freq_left = self.freq
            freq_right = other.freq

            freq_not_in_left = [freq for freq in freq_right if freq not in freq_left]
            freq_not_in_right = [freq for freq in freq_left if freq not in freq_right]

            self.calc(freq_not_in_left)
            other.calc(freq_not_in_right)
        except:
            pass

        return self


    #######################
    #  Helper functions   #
    #######################
    def default_matrix(self, freq=None):
        """
        Creates the default S-matrix per frequency point
        :param freq:    scalar or iterator
        :return:        Return the dataframe
        """

        if freq is None:    # No frequency given: only create an empty DataFrame
            matrix_data = None
        else:
            # First create a flat matrix in one row
            # Start with all zeros (complex valued)
            matrix_flat = np.zeros(self.nports * self.nports, dtype = np.complex_)
            # Create a list of indices of non-zero elements (this is a 'reversed' identity matrix)
            non_zero_col = np.arange(self.nports, 0, -1) - 1 # Need the -1, since it counts down to 1
            row_num = np.arange(0, self.nports, 1)
            idx_non_zero = self.nports * row_num + non_zero_col
            matrix_flat[idx_non_zero] = 1.0
            # Create one row per frequency point
            try:
                n_freq = len(freq)
            except TypeError:
                n_freq = 1
            matrix_data = [matrix_flat for n in range(n_freq)]

        # Create a list of columns (all combination of port numbers
        n_ports = range(1, self.nports+1)
        idx_port = itertools.product(n_ports, n_ports)
        col_names = ['s{:d}{:d}'.format(port[0], port[1]) for port in idx_port]

        # First create the dataframe not containing the frequency variables
        df = pd.DataFrame(data=matrix_data, columns = col_names)

        # Add a column with frequencies
        df['freq'] = freq
        df.set_index('freq', inplace=True)

        return df

    def get_col_name(self, port, par_type='s'):
        """
        Helper function to convert a 2d port number to the correct DataFrame column name, for example: s11, s42, etc
        :param port:    2-component list containing the port numbers.
                        port = (x, y) -> (returns sxy)
        :return:        String of the column name
        """
        return '{}{:d}{:d}'.format(par_type, port[0], port[1])

    def get_col_names(self, par_type='s'):
        n_ports = range(1, self.nports + 1)
        idx_port = itertools.product(n_ports, n_ports)
        col_names = ['{}{:d}{:d}'.format(par_type, port[0], port[1]) for port in idx_port]
        return col_names

    # Add hashable functions to the NPort
    def __str__(self):
        """Print S-Matrix of this port"""
        str_matrix = ''
        # # Convert matrix tp 2D array
        # mtrx_array = self._S.getA()
        # Convert matrix to string row by row
        for freq, series in self._S.iterrows():
            str_matrix += 'Freq = {:.2g}Hz\n'.format(freq)
            for n in range(1, self.nports+1):
                port_list = [self.get_col_name([n, ncol]) for ncol in range(1, self.nports+1)]
                str_list = ['{:8}'.format(series[port_]) for port_ in port_list]
                str_matrix += '\t'.join(str_list) + '\n'

        return str(str_matrix)

    def __hash__(self):
        return hash(str(self))

    # Override default functions
    def __call__(self, freq=None):
        return self.get_s(freq)

    def __invert__(self):
        """Return the inverse of the S-matrix"""
        Sinv = self._S.groupby(self._S.index).apply(lambda x: np.linalg.inv(np.reshape(x)))
        return Sinv

    def __abs__(self):
        """Return the determinant of the matrx"""
        Sabs = self._S.groupby(self._S.index).apply(lambda x: np.linalg.det(x))
        return Sabs

    def __add__(self, other):
        """
        Add other NPort parameters or matrix to this NPort
        :param other:
        :return:
        """
        # Check whether the other is an NPort and has the same # of ports -> if yes then add
        try:
            if other.nports == self.nports:
                # Same number of ports -> add
                df_ret = pd.concat([self._S, other._S])
        except AttributeError:
            # Other does not have nports -> Maybe it is a DataFrame
            try:
                nports = np.sqrt(len(other.columns))
                if nports == self.nports:
                    df_ret = pd.concat([self._S, other])
            except AttributeError:
                raise NotImplementedError('Unable to add {} to Nport'.format(type(other)))

        # Remove double frequencies -> Keep the original row
        df_ret = df_ret[~df_ret.index.duplicated(keep='first')]

        ret = NPort(nports = self.nports, Z0=self.Z0)
        ret._S = df_ret
        return ret

    def __radd__(self, other):
        """
        Add other NPort parameters or matrix to this NPort
        :param other:
        :return:
        """
        return self.__add__(other)

    def __iadd__(self, other):
        """
        Add other NPort parameters or matrix to this NPort
        :param other:
        :return:
        """
        # Check whether the other is an NPort and has the same # of ports -> if yes then add
        try:
            if other.nports == self.nports:
                # Same number of ports -> add
                self._S = pd.concat([self._S, other._S])
        except AttributeError:
            # Other does not have nports -> Maybe it is a DataFrame
            try:
                nports = np.sqrt(len(other.columns))
                if nports == self.nports:
                    self._S = pd.concat([self._S, other])
            except AttributeError:
                raise NotImplementedError('Unable to add {} to Nport'.format(type(other)))

        # Remove double frequencies -> Keep the original row
        self._S = self._S[~self._S.index.duplicated(keep='first')]

        return self

    # Return all the frequencies
    @property
    def freq(self):
        return self._S.index.values

    # Only read the number of ports, can not be changed
    @property
    def nports(self):
        return self.__nports

    # Access S,Z,Y properties
    def get_s(self, freq):
        """
        Return the S parameters for the given frequency.

        :param freq:    Frequency (Hz) of the s parameter data. It can be either a scalar value
                        or a slice. When freq=None all S parameter data will be returned.
        :return:        DataFrame containing the S parameter data.
        """

        # First calculate the new frequency item for all frequencies that were not found
        # Create a set of frequencies already in the dataframe

        if freq is not None:
            # Generate the frequencies if they do not exist yet
            self.generate_freq(freq)
            # # Make a list containing all freqs not in the dataframe
            # freq_present = set(self._S.index.values)
            # try:
            #     freq_req = set(freq)
            #     freq_calc = list(freq_req - freq_present)
            # except TypeError: # freq is not iteratable
            #     if freq not in freq_present:
            #         freq_calc = [freq]
            #     else:
            #         freq_calc = list()
            # # Calculate
            # self.calc(freq_calc)
        else: # When freq = None: return all frequencies
            freq = slice(None)

        # Look up all the frequency points inside the dataframe
        ret = self._S.loc[freq, slice(None)]

        return ret

    def set_s(self, freq, value, spar=None):
        # Create a list of columns (all combination of port numbers
        if spar is None: # When no spar name is given we set all of them
            n_ports = range(1, self.nports+1)
            idx_port = itertools.product(n_ports, n_ports)
            col_names = self.get_col_names('s')
        elif spar not in self._S.columns:
            raise KeyError('Parameter {} is unknown'.format(spar))
        else:
            col_names = spar

        self._S.loc[freq, col_names] = value

        # Concatenate the data
        #self._S = pd.concat([self._S, df])

        return self

    def _S_to_Y(self, data):
        """
        Convert the dataframe containing S parameters into Y parameters
        :param data:
        :return:
        """
        (idx, data) = _to_flat_matrix(data)
        # Reshape to a square 2-dimensional array
        s_matrix = np.reshape(data, (self.nports, self.nports))

        # Create 'help' matrices
        E = np.identity(self.nports)  # Identity matrix
        Yref_sqrt = E / np.sqrt(self.Z0.real)

        # Calculate the Y matrix
        # If E = -self._S than a LinAlgError exception is thrown
        try:
            Y = np.dot(np.dot(Yref_sqrt                  , (E - s_matrix)),
                       np.dot(np.linalg.inv(E + s_matrix), Yref_sqrt))
        except np.linalg.LinAlgError:
            Y = np.ones((self.nports, self.nports)) * np.inf

        # Convert the matrix back to a dataframe
        new_cols = self.get_col_names('y')
        flat_matrix = Y.reshape((1, -1))

        return pd.DataFrame(data = flat_matrix, columns = new_cols, index=idx)

    def _Y_to_S(self, data):
        """
        Convert the dataframe containing Y parameters into S parameters
        :param data:
        :return:
        """
        (idx, data) = _to_flat_matrix(data)
        # Reshape to a square 2-dimensional array
        y_matrix = np.reshape(data, (self.nports, self.nports))

        # Create 'help' matrices
        E = np.identity(self.nports)  # Identity matrix
        Yref_sqrt = E / np.sqrt(self.Z0.real)
        Zref_sqrt = E*np.sqrt(self.Z0.real)

        # Convert to S matrix
        try:
            zyz_matrix = np.dot(np.dot(Zref_sqrt, y_matrix), Zref_sqrt)
            Stmp = np.dot(             (E - zyz_matrix),
                          np.linalg.inv(E + zyz_matrix))
        except np.linalg.LinAlgError:
            Stmp = np.ones((self.nports, self.nports)) * np.inf

        # Convert the matrix back to a dataframe
        new_cols = self.get_col_names('s')
        flat_matrix = Stmp.reshape((1, -1))

        return pd.DataFrame(data = flat_matrix, columns = new_cols, index=idx)

    def _S_to_Z(self, data):
        """
        Convert the dataframe containing S parameters into Z parameters
        :param data:
        :return:
        """
        (idx, data) = _to_flat_matrix(data)
        # Reshape to a square 2-dimensional array
        s_matrix = np.reshape(data, (self.nports, self.nports))

        # Create 'help' matrices
        E = np.matrix( np.identity(self.nports) )       # Identity matrix
        Zref_sqrt = E*np.sqrt(self.Z0.real)

        # Calculate the Z matrix
        # If E = self.S than a LinAlgError exception is thrown
        try:
            Z = np.dot(np.dot(Zref_sqrt     , np.linalg.inv(E - s_matrix)),
                       np.dot((E + s_matrix), Zref_sqrt))
        except np.linalg.LinAlgError:
            Z = np.matrix(np.ones((self.nports, self.nports))) * np.inf

        # Convert the matrix back to a dataframe
        new_cols = self.get_col_names('z')
        flat_matrix = Z.reshape((1, -1))

        return pd.DataFrame(data = flat_matrix, columns = new_cols, index=idx)

    def _Z_to_S(self, data):
        """
        Convert the dataframe containing Z parameters into S parameters
        :param data:
        :return:
        """
        (idx, data) = _to_flat_matrix(data)
        # Reshape to a square 2-dimensional array
        z_matrix = np.reshape(data, (self.nports, self.nports))

        # Create 'help' matrices
        E = np.identity(self.nports)  # Identity matrix
        Yref_sqrt = E / np.sqrt(self.Z0.real)
        Zref_sqrt = E*np.sqrt(self.Z0.real)

        # Convert to S matrix
        try:
            yzy_matrix = np.dot(np.dot(Yref_sqrt, z_matrix), Yref_sqrt)
            Stmp = np.dot(             (yzy_matrix - E),
                          np.linalg.inv(yzy_matrix + E))
        except np.linalg.LinAlgError:
            Stmp = np.ones((self.nports, self.nports)) * np.inf

        # Convert the matrix back to a dataframe
        new_cols = self.get_col_names('s')
        flat_matrix = Stmp.reshape((1, -1))

        return pd.DataFrame(data = flat_matrix, columns = new_cols, index=idx)

    def get_parameter(self, freq=None, par_type='ABCD'):
        '''
        Get the parameters of the given type.
        :param freq: Frequencies to return, can be a Slice
        :param par_type: Type of parameter to return
                    All ports: Z/Y
                    Only 2-ports: T/ABCD
        :return: parameters as pd.DataFrame
        '''
        # Get S parameters
        sdata = self.get_s(freq)
        if par_type.lower() == 's':
            # If S parameters are asked -> return them
            return sdata

        # If the frequency does not exist (empty dataframe) -> return an empty dataframe
        if len(sdata) == 0:
            return pd.DataFrame(columns=self.get_col_names(par_type.lower()), index=sdata.index)

        # Get parameter convert function
        func_convert = getattr(self, '_S_to_{}'.format(par_type.upper()))

        if isinstance(sdata, pd.DataFrame):
            new_data = sdata.groupby(sdata.index).apply(func_convert)
        else:
            new_data = func_convert(sdata)

        return new_data

    def set_parameter(self, freq=None, value=None, par_type='Y', par_name=None):
        # If we want to set S-Parameters call that function
        if par_type.lower() == 's':
            self.set_s(freq, value, spar=par_name)
            return self

        # Not S-parameters
        col_names = self.get_col_names(par_type.lower())
        # Create a list of columns (all combination of port numbers
        if par_name in col_names:
            col_names = par_name
        elif par_name is None:
            pass
        else:
            raise KeyError('Parameter {} is unknown'.format(par_name))

        # Get parameter convert function
        func_convert_from_s = getattr(self, '_S_to_{}'.format(par_type.upper()))
        func_convert_to_s = getattr(self, '_{}_to_S'.format(par_type.upper()))

        if isinstance(value, pd.Series):
            # Convert to S parameters
            sdata = func_convert_to_s(value[col_names])
            # Add s parameter data
            self._S = pd.concat([self._S, sdata], ignore_index=False)
            # Remove duplicates
            self._S = self._S[~self._S.index.duplicated(keep='last')]
        elif isinstance(value, pd.DataFrame):
            # Convert to S parameters
            sdata = value.groupby(value.index).apply(func_convert_to_s)
            # Add s parameter data
            self._S = pd.concat([self._S, sdata], ignore_index=False)
            # Remove duplicates
            self._S = self._S[~self._S.index.duplicated(keep='last')]
        else: # isinstance(value, np.ndarray) or isinstance(value, list):
            # Get S parameters -> convert to chosen parameters -> Set col -> Convert back
            sdata = self.get_s(freq)
            new_data = func_convert_from_s(sdata)
            new_data[col_names] = np.reshape(value, (1, -1)).astype(np.complex_)
            sdata = func_convert_to_s(new_data)
            try:
                self._S.loc[freq] = sdata.values
            except TypeError:
                # We are trying to set a nport=1 while we should set a scalar
                self._S.loc[freq] = sdata.values.flatten()[0]

        return self


##########################
#      Main function     #
##########################
if __name__ == '__main__':
    # Unit test
    Port0 = NPort(nports=2)
    Port0.s[1e9]
    Port0.s11[1e9] = 0.01
    Port0.s22[1e9] = 0.01
    Port0.s12[1e9] = 0.00
    Port0.s21[1e9] = 0.00
    print(Port0.s[1e9])
    print(Port0.s[None])

    print('\nSet y11')
    print(Port0.y[1e9])
    Port0.set_parameter(freq = 1e9, value = 1234, par_type='y', par_name='y11')
    print(Port0.y[1e9])

    print('\nSet z11')
    print(Port0.z[1e9])
    Port0.set_parameter(freq = 1e9, value = 1234, par_type='z', par_name='z11')
#    Port0.set_z(1e9, 1234, 'z11')
    print(Port0.z[1e9])

    # Try to get address the frequencies
    print('\nTest all the s-parameter __getitem__ functions')
    print('List of all frequencies: {}'.format(Port0.freq))
    Port0.s11[0]
    Port0.s11[2e9]
    Port0.s11[1e9]
    Port0.s12[1e9]
    Port0.s21[1e9]
    Port0.s22[1e9]
    print('\nList of all frequencies: {}'.format(Port0.freq))

    print('\nTest all the s-parameter __setitem__ functions')
    Port0.s11[0] = 101.0

    print('\nTest __getitem__ with a scalar and with a list')
    Port0.s11[0]
    Port0.s11[[0, 1e9]]

    print('\nTest the iterator of the s11 parameters:')
    for n, sdata in enumerate(Port0.s11):
        print('\tItem {:d}: {}'.format(n, sdata))

    print('\nTest the iterator of the s parameter matrix:')
    for n, sdata in enumerate(Port0.s):
        print('\tItem {:d}:\n{}'.format(n, sdata))

    print('\nTest the iterator of the y parameter matrix:')
    for n, sdata in enumerate(Port0.y):
        print('\tItem {:d}:\n{}'.format(n, sdata))

    print('\nSet s12=s21=0, this opens both sides, and z11=z22=Z0')
    Port0.s11[1e9] = 0.0001
    Port0.s21[1e9] = 0
    Port0.s12[1e9] = 0
    print('\nPrint the dataframe returned by a call to the class:')
    print(str(Port0))
