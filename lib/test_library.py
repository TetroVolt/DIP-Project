import unittest as test
import numpy as np

import lib

from typing import Tuple

class LibraryTest(test.TestCase):

    def assert_array(self, expected: np.array, actual: np.array) -> Tuple[bool, str]:
        """
            Checks that the sizes and contents of the two provided arrays are the same.

            Args:
                expected (:class: array):  The values we expected to see.

                actual (:class: array):  The values we did received.

            Returns:
                (:class: tuple):  A tuple containing True/False for if we passed, and if False, a message
                    clarifying the nature of the failure.
        """
        expected_size = expected.shape
        actual_size = actual.shape
        if expected_size != actual_size:
            return (False, "Arrays are not the same size.  Should be {} is {}.".format(expected_size, actual_size))
        rows, columns = expected_size
        for row in range(0, rows):
            for column in range(0, columns):
                exp_val = expected[row, column]
                actual_val = actual[row, column]
                if exp_val != actual_val:
                    return (False, "Array contents are different.  Should be {} is {}.".format(exp_val, actual_val))
        return (True, "")

    def setUp(self):
        self.rows, self.columns = (3, 3)
        self.input_matrix = np.zeros((self.rows, self.columns), dtype=np.uint8)
        temp = 1
        # Create an input matrix filled with 1-9
        for row in range(0, self.rows):
            for column in range(0, self.columns):
                self.input_matrix[row, column] = temp
                temp += 1

    def test_resize(self):
        expected_downsize = np.zeros((2, 2), dtype=np.uint8)
        expected_downsize[0] = [4, 4]
        expected_downsize[1] = [5, 6]
        result = lib.resize(self.input_matrix, (2, 2))
        truth, msg = self.assert_array(expected_downsize, result)
        self.assertTrue(truth, "Downsizing was incorrect. {}".format(msg))

    def test_warpAffine(self):
        #TODO-Add assertions on expected output.
        transformation = lib.getRotationMatrix2D((3/2, 3/2), 90, 1)
        lib.warpAffine(self.input_matrix, transformation, (3, 3))

    def test_rotation2D(self):
        rows, columns = self.input_matrix.shape

        alpha = 6.123233995736766e-17
        beta = 1.0

        # Expected returned values for 90 degrees and center.
        expected = np.zeros((2, 3), dtype=np.float)
        expected[0] = [alpha, beta, -2.220446049250313e-16]
        expected[1] = [-beta, alpha, 3.0]

        result = lib.getRotationMatrix2D((rows/2, columns/2), 90, 1)
        truth, msg = self.assert_array(expected, result)
        self.assertTrue(truth, "3, 3 matrix was invalid. {}".format(msg))

        # Ensure a tiny matrix gets the same result with the same paramaters.
        self.input_matrix = np.zeros((1,1), dtype=float)
        rows, columns = self.input_matrix.shape

        expected[0] = [alpha, beta, -5.551115123125783e-17]
        expected[1] = [-beta, alpha, 1.0]

        result = lib.getRotationMatrix2D((rows/2, columns/2), 90, 1)
        truth, msg = self.assert_array(expected, result)
        self.assertTrue(truth, "1, 1 matrix was invalid. {}".format(msg))

    def test_shear_transform(self):
        #TODO-Add assertions on expected output.
        lib.shear_transform(self.input_matrix, 0)

    def test_get_perspective_transform(self):
        #TODO-Add assertions on expected output.
        lib.perspective_transform(self.input_matrix, 0)

    def test_warp_perspective(self):
        #TODO-Add assertions on expected output.
        lib.warp_perspective(self.input_matrix, 0, 0, 0, 0, 0, 0)

    def test_exports(self):
        exports = lib.get_exports()
        blacklist = ["np", "get_exports", "test_library"]
        # Get all the functions and attributes of the module, and filter by only the public functions.
        function_names = [attr for attr in dir(lib) if "__" not in attr and attr not in blacklist]
        functions = []
        for key in exports:
            index = function_names.index(key)
            # Ensure that the function is the same as it's key name.
            self.assertTrue(exports[key] == getattr(lib, function_names.pop(index)),
                            msg="Key is not the same as it's function.")

        # Make sure all of the public functions were in the dictionary.  If not, assert.
        self.assertFalse(functions, msg="Not enough keys for all public functions.")

