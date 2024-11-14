import unittest
import numpy as np
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cost_functions import (
    l1_error_function,
    l2_error_function,
    cross_entropy_loss_error_function,
    mean_squared_error,
    mean_absolute_error,
    binary_crossentropy,
    hinge_loss_error_function
)

class TestCostFunctions(unittest.TestCase):

    def setUp(self):
        self.y = np.array([1, 0, 1, 0])
        self.y_pred = np.array([0.9, 0.1, 0.8, 0.2])

    def test_l1_error_function(self):
        result = l1_error_function(self.y, self.y_pred)
        expected = np.sum(np.abs(self.y - self.y_pred))
        self.assertAlmostEqual(result, expected)

    def test_l2_error_function(self):
        result = l2_error_function(self.y, self.y_pred)
        expected = np.sum(np.power(self.y - self.y_pred, 2))
        self.assertAlmostEqual(result, expected)

    def test_cross_entropy_loss_error_function(self):
        result = cross_entropy_loss_error_function(self.y, self.y_pred)
        expected = -np.sum(self.y * np.log(self.y_pred) + (1 - self.y) * np.log(1 - self.y_pred))
        self.assertAlmostEqual(result, expected)

    def test_mean_squared_error(self):
        result = mean_squared_error(self.y, self.y_pred)
        expected = np.mean(np.power(self.y - self.y_pred, 2))
        self.assertAlmostEqual(result, expected)

    def test_mean_absolute_error(self):
        result = mean_absolute_error(self.y, self.y_pred)
        expected = np.mean(np.abs(self.y - self.y_pred))
        self.assertAlmostEqual(result, expected)

    def test_binary_crossentropy(self):
        result = binary_crossentropy(self.y, self.y_pred)
        expected = np.mean(-self.y * np.log(self.y_pred) - (1 - self.y) * np.log(1 - self.y_pred))
        self.assertAlmostEqual(result, expected)

    def test_hinge_loss_error_function(self):
        result = hinge_loss_error_function(self.y, self.y_pred)
        expected = np.mean(np.maximum(0, 1 - self.y * self.y_pred))
        self.assertAlmostEqual(result, expected)

if __name__ == '__main__':
    unittest.main()