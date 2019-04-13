import unittest
import numpy

import lib

class LibraryTest(unittest.TestCase):

    def setUp(self):
        self.rows, self.columns = (3, 3)
        self.input_matrix = numpy.zeros((self.rows, self.columns, 3), dtype=numpy.uint8)
        temp = 1
        # Create an input matrix filled with 1-9
        for row in range(0, self.rows):
            for column in range(0, self.columns):
                self.input_matrix[row, column] = temp
                temp += 1

    def test_resize(self):
        # TODO-Add assertions on expected output.
        lib.resize(self.input_matrix, 0)

    def test_warp_affine(self):
        #TODO-Add assertions on expected output.
        lib.warp_affine(self.input_matrix, 0, 0, 0, 0, 0, 0)

    def test_rotation_2D(self):
        #TODO-Add assertions on expected output.
        lib.get_rotation_matrix_2D(0, 0, 0, 0)

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

