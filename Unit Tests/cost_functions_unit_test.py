import unittest
import numpy as np
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../main')))


from activation_functions import (
    linear_activation, binary_step_activation, ReLU_activation, PReLU_activation,
    ELU_activation, SLU_activation, softplus_activation, logistic_activation,
    tanh_activation, arctan_activation, linear_activation_gradient,
    binary_step_activation_gradient, ReLU_activation_gradient,
    PReLU_activation_gradient, ELU_activation_gradient,
    SLU_activation_gradient, logistic_activation_gradient,
    softplus_activation_gradient, tanh_activation_gradient,
    arctan_activation_gradient
)

class TestActivationFunctions(unittest.TestCase):

    def test_linear_activation(self):
        self.assertEqual(linear_activation(5), 5)
        self.assertEqual(linear_activation(-3), -3)

    def test_binary_step_activation(self):
        self.assertEqual(binary_step_activation(5), 1)
        self.assertEqual(binary_step_activation(-3), 0)

    def test_ReLU_activation(self):
        self.assertEqual(ReLU_activation(5), 5)
        self.assertEqual(ReLU_activation(-3), 0)

    def test_PReLU_activation(self):
        self.assertEqual(PReLU_activation(5), 5)
        self.assertEqual(PReLU_activation(-3), -0.03)
        self.assertEqual(PReLU_activation(-3, alpha=0.1), -0.3)

    def test_ELU_activation(self):
        self.assertEqual(ELU_activation(5), 5)
        self.assertAlmostEqual(ELU_activation(-3), -0.95021293)
        self.assertAlmostEqual(ELU_activation(-3, alpha=0.5), -0.47510646)

    def test_SLU_activation(self):
        self.assertAlmostEqual(SLU_activation(5), 4.966527196)
        self.assertAlmostEqual(SLU_activation(-3), -0.952574127)

    def test_softplus_activation(self):
        self.assertAlmostEqual(softplus_activation(5), 5.006715348)
        self.assertAlmostEqual(softplus_activation(-3), 0.048587351)

    def test_logistic_activation(self):
        self.assertAlmostEqual(logistic_activation(5), 0.993307149)
        self.assertAlmostEqual(logistic_activation(-3), 0.047425873)

    def test_tanh_activation(self):
        self.assertAlmostEqual(tanh_activation(5), 0.999909204)
        self.assertAlmostEqual(tanh_activation(-3), -0.995054754)

    def test_arctan_activation(self):
        self.assertAlmostEqual(arctan_activation(0), 1.373400767)
        self.assertAlmostEqual(arctan_activation(-3), -1.249045772)

    def test_linear_activation_gradient(self):
        self.assertEqual(linear_activation_gradient(5), 1)
        self.assertEqual(linear_activation_gradient(-3), 1)

    def test_binary_step_activation_gradient(self):
        self.assertEqual(binary_step_activation_gradient(5), 0)
        self.assertEqual(binary_step_activation_gradient(-3), 0)

    def test_ReLU_activation_gradient(self):
        self.assertEqual(ReLU_activation_gradient(5), 1)
        self.assertEqual(ReLU_activation_gradient(-3), 0)

    def test_PReLU_activation_gradient(self):
        self.assertEqual(PReLU_activation_gradient(5), 1)
        self.assertEqual(PReLU_activation_gradient(-3), 0.01)
        self.assertEqual(PReLU_activation_gradient(-3, alpha=0.1), 0.1)

    def test_ELU_activation_gradient(self):
        self.assertEqual(ELU_activation_gradient(5), 1)
        self.assertAlmostEqual(ELU_activation_gradient(-3), 0.049787068)
        self.assertAlmostEqual(ELU_activation_gradient(-3, alpha=0.5), 0.024893534)

    def test_SLU_activation_gradient(self):
        self.assertAlmostEqual(SLU_activation_gradient(5), 1.02654743243)
        self.assertAlmostEqual(SLU_activation_gradient(-3), 0.002466509)

    def test_logistic_activation_gradient(self):
        self.assertAlmostEqual(logistic_activation_gradient(5), 0.006648056)
        self.assertAlmostEqual(logistic_activation_gradient(-3), 0.002466509)

    def test_softplus_activation_gradient(self):
        self.assertAlmostEqual(softplus_activation_gradient(5), 0.993307149)
        self.assertAlmostEqual(softplus_activation_gradient(-3), 0.047425873)

    def test_tanh_activation_gradient(self):
        self.assertAlmostEqual(tanh_activation_gradient(5), 0.000181583)
        self.assertAlmostEqual(tanh_activation_gradient(-3), 0.009866037)

    def test_arctan_activation_gradient(self):
        self.assertAlmostEqual(arctan_activation_gradient(0), 1.0)
        self.assertAlmostEqual(arctan_activation_gradient(-3), 0.1)

if __name__ == '__main__':
    unittest.main()