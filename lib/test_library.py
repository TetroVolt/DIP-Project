import unittest
import lib
import numpy

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
        resize(0, 0)

    def test_warp_affine(self):
        # This needs to be updated once the warp_affine is finished.
        library.warp_affine(self.input_matrix, 0, 0, 0, 0, 0, 0)

    def test_rotation_2D(self):
        pass
    def test_shear_transform(self):
        pass
    def test_resize(self):
        pass

