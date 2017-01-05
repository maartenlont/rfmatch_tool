from unittest import TestCase
import random
import numpy as np
import pandas as pd
import itertools

from NPort import _to_matrix


class Test_to_matrix(TestCase):
    def setUp(self):
        random.seed()
        # Randomly set the number of ports
        self.nports = random.randint(2, 11)

        # Create a valid DataFrame
        # randomly create the matrix data
        self.data_ok = [1e3*(random.random()-0.5) for n in range(self.nports*self.nports)]
        self.df_ok = pd.DataFrame(data=self.data_ok)

        # Create an invalid DataFrame
        self.data_nok = [1e3*(random.random()-0.5) for n in range(self.nports*self.nports+1)]
        self.df_nok = pd.DataFrame(data=self.data_nok)

    def test_convert_data_ok(self):
        # Convert the dataframe
        data_matrix = _to_matrix(self.df_ok)

        # Check whether the data is correct
        for n, (port_r, port_c) in enumerate(itertools.product(range(self.nports), range(self.nports))):
            self.assertEqual(data_matrix[port_r, port_c], self.data_ok[n], 'Matrix not equal for {:d},{:d}'.format(port_r, port_c))

    def test_convert_num_dimensions(self):
        # Convert the dataframe
        data_matrix = _to_matrix(self.df_ok)

        # Check the size of the matrix
        matrix_size = np.shape(data_matrix)
        self.assertEqual(len(matrix_size), 2, 'Returned matrix is not 2-Dimensional')

    def test_convert_dimensions(self):
        # Convert the dataframe
        data_matrix = _to_matrix(self.df_ok)

        # Check the size of the matrix
        matrix_size = np.shape(data_matrix)
        self.assertEqual(matrix_size[0], self.nports,
                         'Matrix 1st dimension ({:d}) is not correct ({:d})'.format(matrix_size[0], self.nports))
        self.assertEqual(matrix_size[1], self.nports,
                         'Matrix 2nd dimension ({:d}) is not correct ({:d})'.format(matrix_size[1], self.nports))

    def test_convert_data_nok(self):
        # Convert the dataframe
        with self.assertRaises(ValueError, msg='No exception raised when data is not "square"'):
            data_matrix = _to_matrix(self.df_nok)
