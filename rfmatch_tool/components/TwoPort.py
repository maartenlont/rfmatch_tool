import numpy as np
import pandas as pd

from .NPort import NPort, DataDescriptor, DataMatrixDescriptor, _to_flat_matrix
from .OnePort import OnePort
import itertools

class TwoPort(NPort):
    def __init__(self, freq=None, Z0=50., S=None):
        # Create a special case of the NPort, namly only two ports
        super(TwoPort, self).__init__(freq=freq, nports=2, Z0=Z0, S=S)

        # Create list of ports
        ports = range(1, self.nports+1)
        ports_comb = itertools.product(ports, ports)

        # Create functions that return the t parameters
        for par_type_ , port_ in itertools.product(['t'], ports_comb):
            name = self.get_col_name(port_, par_type=par_type_)
            self.__dict__[name] = DataDescriptor(self, name)
            self.parameter_list += [name]

        # Get abcd and T parameter matrix
        #self.abcd = DataMatrixDescriptor(self, 'abcd')
        self.__dict__['abcd'] = DataMatrixDescriptor(self, 'abcd')
        self.__dict__['t'] = DataMatrixDescriptor(self, 't')

    # Ocerload the NPort get_col_names for the ABCD case
    def get_col_names(self, par_type='s'):
        if par_type == 'abcd':
            col_names = ['A', 'B', 'C', 'D']
        else:
            # Not abcd parameters -> Use defaults
            col_names = super(TwoPort, self).get_col_names(par_type)
        return col_names

    def _S_to_ABCD(self, data):
        """
        Convert the dataframe containing S parameters into abcd parameters. The index is kept when either a
        DataFrame or Series is given.
        :param data: DataFrame / Series / 2D array (matrix) containing S parameters
        :return: DataFrame with ABCD parameters (columns)
        """
        (idx, data) = _to_flat_matrix(data)
        # Unpack the parameters
        (s11, s12, s21, s22) = data

        A =             ((1+s11)*(1-s22) + s12*s21) / (2*s21)
        B =   self.Z0 * ((1+s11)*(1+s22) - s12*s21) / (2*s21)
        C = 1/self.Z0 * ((1-s11)*(1-s22) - s12*s21) / (2*s21)
        D =             ((1-s11)*(1+s22) + s12*s21) / (2*s21)
        data_flat = [[A, B, C, D]]

        return pd.DataFrame(data = data_flat, columns = ['A', 'B', 'C', 'D'], index=idx)

    def _ABCD_to_S(self, data):
        """
        Convert the dataframe containing ABCD into S parameters. The index is kept when possible
        :param data:
        :return:
        """
        (idx, data) = _to_flat_matrix(data)
        # Unpack the parameters
        (A, B, C, D) = data

        S11 = ( A + B/self.Z0 - C*self.Z0 - D) / ( A + B/self.Z0 + C*self.Z0 + D )
        S12 = ( 2*(A*D-B*C)                  ) / ( A + B/self.Z0 + C*self.Z0 + D )
        S21 = ( 2                            ) / ( A + B/self.Z0 + C*self.Z0 + D )
        S22 = (-A + B/self.Z0 - C*self.Z0 + D) / ( A + B/self.Z0 + C*self.Z0 + D )
        data_flat = [[S11, S12, S21, S22]]

        return pd.DataFrame(data = data_flat, columns = self.get_col_names('s'), index=idx)

    def _S_to_T(self, data):
        """
        Convert the dataframe containing S parameters into abcd parameters. The index is kept when either a
        DataFrame or Series is given.
        :param data: DataFrame / Series / 2D array (matrix) containing S parameters
        :return: DataFrame with ABCD parameters (columns)
        """
        (idx, data) = _to_flat_matrix(data)
        # Unpack the parameters
        (s11, s12, s21, s22) = data

        T11 = ( s12*s21 - s11*s22 ) / s21
        T12 = s11 / s21
        T21 = - s22 / s21
        T22 = 1 / s21
        data_flat = [[T11, T12, T21, T22]]

        return pd.DataFrame(data = data_flat, columns = self.get_col_names('t'), index=idx)

    def _T_to_S(self, data):
        """
        Convert the dataframe containing ABCD into S parameters. The index is kept when possible
        :param data:
        :return:
        """
        (idx, data) = _to_flat_matrix(data)
        # Unpack the parameters
        (T11, T12, T21, T22) = data

        S11 = T12 / T22
        S12 = (T11*T22 - T12*T21) / T22
        S21 = 1 / T22
        S22 = - T21 / T22
        data_flat = [[S11, S12, S21, S22]]

        return pd.DataFrame(data = data_flat, columns = self.get_col_names('s'), index=idx)

    # Overide math operators
    def _abcd_matrix_multiply(self, group):
        """
        Multiply two ABCD matrices. The left/right matrix should have a '_l'/'_r' suffix.
        :param group: DataFrame containing the left and right matrices.
        :return: DataFrame containing the resulting matrix
        """
        # Get right and left matrices
        abcd_left = group['matrix_l'].values[0]
        abcd_right = group['matrix_r'].values[0]

        # Multiply the matrices
        matrix_new = np.dot(abcd_left, abcd_right)

        # Return the new matrix
        return pd.DataFrame({'A': matrix_new[0, 0],
                             'B': matrix_new[0, 1],
                             'C': matrix_new[1, 0],
                             'D': matrix_new[1, 1]}, index = group.index)

    def _abcd_oneport_multiply(self, group):
        """
        Multiply an ABCD matrix and a vector. The left matrix should have an '_l' suffix and the
        the vector should have an '_r' suffix.
        :param group: DataFrame containing the left matrix 'matrix_l' and right vector 'vector_r'.
        :return: DataFrame containing the resulting matrix
        """
        # Get right and left matrices
        abcd_left = group['matrix'].values[0]
        vector_right = group['vector'].values[0]

        # Multiply the matrices
        vector_new = np.dot(abcd_left, vector_right)
        with np.errstate(divide='ignore'):
            z11 = vector_new[0] / vector_new[1]

        # Return the new matrix
        return pd.DataFrame({'z11': z11[0], 'vector': [vector_new]}, index = group.index)

    def __mul__(self, other):
        """
        Multiply two TwoPorts or one TwoPort and one OnePort: place them in series and return
        """
        ret = TwoPort()
        # Synchronize the frequency list of the two NPorts
        self.sync_freq(other)

        # First try to multiply this TwoPort by the other TwoPort
        try:
            # Get left and right matrices
            abcd_left = self.abcd[None]
            abcd_right = other.abcd[None]
            # Merge into one dataframe and apply multiplication
            abcd_both = abcd_left.merge(abcd_right, left_index=True, right_index=True, suffixes=('_l', '_r'))
            abcd_new = abcd_both.groupby(abcd_both.index).apply(self._abcd_matrix_multiply)
            # Store final result
            ret.abcd.__set__(ret, abcd_new)
        except AttributeError: # Other component does not have abcd parameters : not a twoport-like object
            # Try to multiply with a OnePort
            try:
                # Calculate one-port vector
                # Get left and right matrices
                abcd_left = self.abcd[None]
                vector_right = other.vector(None)

                # Merge into one dataframe and apply multiplication
                abcd_both = abcd_left.merge(vector_right, left_index=True, right_index=True)
                vector_new = abcd_both.groupby(abcd_both.index).apply(self._abcd_oneport_multiply)
                # Store final result
                ret = OnePort(Z0=self.Z0)
                ret.z.__set__(ret, vector_new['z11'])
            except: # Other component does not have a vector in the right format
                # Try to multiply a scalar
                try:
                    ret.S = self._S * other
                except:
                    raise ValueError('Unknown type {}'.format(type(other)))
                    ret.S = self._S


        return ret

if __name__ == '__main__':
    Port1 = TwoPort(freq=1e9)
    print(Port1)
    Port2 = TwoPort()
    Port3 = TwoPort()

    print(Port1.abcd[None])
    Zser = 0.; Z0=50.; Zpar = 50.
    Port1.abcd[1e6]=np.array([[1, Zser], [0, 1]])
    print(Port1)
    print(Port1.y[1e6])
    Port2.abcd[1e6]=np.array([[1, 0], [1/Zpar, 1]])

    print('\n** Multiply')
    print(Port1.abcd)
    print('** by')
    print(Port2.abcd)
    print('** Makes')
    Port3 = Port1 * Port2
    print(Port3.abcd)

    print('\n** Test multiplication of twoport and oneport')
    load = OnePort()
    load.z11[0] = 10.0
    load.z11[1e5] = 15.0
    load.z11[1e6] = 20.0
    Port1.abcd[0] = np.array([[1, 0], [0, 1]]) #np.matrix('1, 0;0, 1')
    print((Port1 * load).z11[None])
