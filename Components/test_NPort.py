from unittest import TestCase
import numpy as np
import random

from NPort import NPort


class TestNPort(TestCase):
    def setUp(self):
        self.Z0 = 50.0

        # Create S parameter matrix of through connection. Note that s11=s22 > 0 to be able
        # to convert to Y and Z
        self.port_through = NPort(0)
        self.port_through.s11[0] = 1e-10
        self.port_through.s22[0] = 1e-10

        # Create S parameter matrix of short connection. Note that s11=s22 > 0 to be able
        # to convert to Y and Z. s21=s12=0 (not connected.
        self.port_short = NPort(0)
        self.port_short.s11[0] = 1e-10
        self.port_short.s22[0] = 1e-10
        self.port_short.s12[0] = 0.0
        self.port_short.s21[0] = 0.0

        # Y and Z parameters of the short
        self.Y_short = np.array([[1/self.Z0, 0], [0, 1/self.Z0]])
        self.Z_short = np.array([[  self.Z0, 0], [0,   self.Z0]])

        # Create random port with random # of ports. Used to test the get_col_names and get_nports
        self.nport_rand = random.randint(2, 10)
        self.port_random = NPort(freq = 0, nports=self.nport_rand)

    def test_S_Y_S(self):
        """
        Convert the S parameters to Y and back again. The resulting matrix should be the same
        """
        for port_ in [self.port_short, self.port_through]:
            S0 = port_.s[0]
            Y  = port_._S_to_Y(S0)
            S1 = port_._Y_to_S(Y)

            # Reshape and compare S0 and S1
            S0_flat = np.reshape(S0, (1,-1))
            S1_flat = S1.values
            S_equal = np.allclose(S0_flat, S1_flat)

            self.assertTrue(S_equal, 'S->Y->S did not return the same matrix')

    def test_S_Z_S(self):
        """
        Convert the S parameters to Z and back again. The resulting matrix should be the same
        """
        for port_ in [self.port_short, self.port_through]:
            S0 = port_.s[0]
            Z  = port_._S_to_Z(S0)
            S1 = port_._Z_to_S(Z)

            # Reshape and compare S0 and S1
            S0_flat = np.reshape(S0, (1,-1))
            S1_flat = S1.values
            S_equal = np.allclose(S0_flat, S1_flat)

            self.assertTrue(S_equal, 'S->Z->S did not return the same matrix')

    def test_nports(self):
        self.assertEqual(self.nport_rand, self.port_random.nports, 'Returned number of ports is not correct')

    def test__S_to_Y(self):
        # test the short port
        Y_short = self.port_short.y[0]
        Y_short = np.reshape(Y_short.values[0], (1, -1))
        Y_short_OK = np.allclose(Y_short, np.reshape(self.Y_short, (1, -1)))
        self.assertTrue(Y_short_OK, 'S->Y of short is incorrect\n{}\n{}'.format(Y_short, self.Y_short))

    def test__Y_to_S(self):
        S_short = self.port_short._Y_to_S(self.Y_short)
        S_short = S_short.values
        S_short_port = np.reshape(self.port_short.s[0], (1, -1))
        S_short_OK = np.allclose(S_short, S_short_port)
        self.assertTrue(S_short_OK, 'Y->S of short is incorrect\n{}\n{}'.format(S_short, S_short_port))

    def test__S_to_Z(self):
        # test the short port
        Z_short = self.port_short.z[0]
        Z_short = np.reshape(Z_short.values[0], (1, -1))
        Z_short_OK = np.allclose(Z_short, np.reshape(self.Z_short, (1, -1)))
        self.assertTrue(Z_short_OK, 'S->Z of short is incorrect\n{}\n{}'.format(Z_short, self.Z_short))

    def test__Z_to_S(self):
        S_short = self.port_short._Z_to_S(self.Z_short)
        S_short = S_short.values
        S_short_port = np.reshape(self.port_short.s[0], (1, -1))
        S_short_OK = np.allclose(S_short, S_short_port)
        self.assertTrue(S_short_OK, 'Z->S of short is incorrect\n{}\n{}'.format(S_short, S_short_port))

    def test_set_y(self):
        # Set a random port and read back
        self.fail()