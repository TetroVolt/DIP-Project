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
        lib.warpAffine(self.input_matrix, transformation, (3, 3), lib.INTER_LINEAR)

    def test_warpAffine_badpath(self):
        transformation = lib.getRotationMatrix2D((3/2, 3/2), 90, 1)
        with self.assertRaisesRegex(ValueError, "Transform Matrix is the incorrect size."):
            lib.warpAffine(self.input_matrix, np.float32([[1,1]]), (3, 3), lib.INTER_NEAREST)

    def test_getRotationMatrix2D(self):
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

    def test_shearTransform(self):
        #TODO-Add assertions on expected output.
        lib.shearTransform(self.input_matrix, self.input_matrix)

    def test_getPerspectiveTransform(self):
        srcPoints = np.float32([[56,65], [368,52], [28,387], [389,390]])
        dstPoints = np.float32([[0,0], [300,0], [0,300], [300,300]])
        expected = np.float64([[1.0558737612964193,  0.09181510967794954, -65.0969127616662],
                              [0.046901004937543685,  1.1256241185010436, -75.79202397907027],
                              [0.00018325144816951961,  0.0005133370013304123,  1.0]])

        result = lib.getPerspectiveTransform(srcPoints, dstPoints)
        truth, msg = self.assert_array(expected, result)
        self.assertTrue(truth, "Transform Matrix was not the same. {}".format(msg))

    def test_getPerspectiveTransform_badpath(self):
        expected_message = "There must be four source points and four destination points."
        with self.assertRaisesRegex(ValueError, expected_message):
            src = np.float32([[1,4], [5, 6]])
            dst = np.float32([[1,1], [2,2]])
            lib.getPerspectiveTransform(src, dst)

    def test_warpPerspective(self):
        #TODO-Add assertions on expected output.
        srcPoints = np.float32([[56,65], [368,52], [28,387], [389,390]])
        dstPoints = np.float32([[0,0], [300,0], [0,300], [300,300]])

        transform = lib.getPerspectiveTransform(srcPoints, dstPoints)
        lib.warpPerspective(self.input_matrix, transform, (500, 500))

    def test_fisheye(self):
        lib.fisheye(self.input_matrix)

    def test_exports(self):
        exports = lib.get_exports()
        blacklist = ["np", "math", "Tuple", "get_exports", "test_library", "INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC",
                     "BORDER_CONSTANT", "BORDER_REPLICATE"]
        # Get all the functions and attributes of the module, and filter by only the public functions.
        function_names = [attr for attr in dir(lib) if "__" not in attr and attr not in blacklist]
        for key in exports:
            index = function_names.index(key)
            # Ensure that the function is the same as it's key name.
            self.assertTrue(exports[key] == getattr(lib, function_names.pop(index)),
                            msg="Key is not the same as it's function.")

        # Make sure all of the public functions were in the dictionary.  If not, assert.
        self.assertFalse(function_names, msg="Not enough keys for all public functions.")

