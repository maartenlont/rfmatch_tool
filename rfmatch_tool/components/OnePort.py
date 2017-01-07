import numpy as np
import pandas as pd
from .NPort import NPort, DataMatrixDescriptor, _to_flat_matrix

class OnePort(NPort):
    def __init__(self, freq=None, Z0=50., S=None):
        # Create a special case of the NPort, namly only two ports
        super(OnePort, self).__init__(freq=freq, nports=1, Z0=Z0, S=S)

        # Create Gamma property
        self.__dict__['gamma'] = DataMatrixDescriptor(self, 'gamma')
        self.parameter_list += ['gamma']

    def get_col_names(self, par_type='s'):
        if par_type == 'gamma':
            col_names = ['gamma']
        else:
            # Not abcd parameters -> Use defaults
            col_names = super(OnePort, self).get_col_names(par_type)
        return col_names

    def _S_to_GAMMA(self, data):
        """
        Convert the dataframe containing S parameters into gamma parameters. The index is kept when either a
        DataFrame or Series is given.
        :param data: DataFrame / Series / 2D array (matrix) containing S parameters
        :return: DataFrame with GAMMA parameter (column)
        """
        # Data contains only s11
        (idx, data) = _to_flat_matrix(data)

        return pd.DataFrame(data = data, columns = self.get_col_names('gamma'), index=idx)

    def _GAMMA_to_S(self, data):
        """
        Convert the dataframe containing ABCD into S parameters. The index is kept when possible
        :param data:
        :return:
        """
        (idx, data) = _to_flat_matrix(data)

        return pd.DataFrame(data = data, columns = self.get_col_names('s'), index=idx)

    # Properties
    #   RL
    #   VSWR
    @property
    def RL(self):
        return -20*np.log10(abs(self.s11))
    @property
    def VSWR(self):
        return (1+abs(self.s11[None])) / (1-abs(self.s11[None]))

    def _convert_to_vector(self, group):
        # Data contains only s11
        (idx, x) = _to_flat_matrix(group)
        vector = np.array([[1], [1/x[0]]]).astype(np.complex_)
        return pd.DataFrame({'vector': [vector]}, index=idx)

    def vector(self, freq):
        zdata = self.z[freq]
        vec = zdata.groupby(zdata.index)['z11'].apply(self._convert_to_vector)
        return vec

if __name__ == '__main__':
    # TODO: Unit test Z11 and Y11 getter and setter
    Port1 = OnePort()
    print(Port1)
    Port1.z11[0] = 10
    Port1.z11[1] = 11
    print(Port1.z11[None])

    print('\nComplex gamma test')
    Port1.gamma[20] = complex(0.1, 1)
    print(Port1.gamma)
    print(Port1)

    print('\nZ parameters')
    print(Port1.z11[None])
    print(Port1.VSWR)

    print('\nVector test')
    print(Port1.vector())