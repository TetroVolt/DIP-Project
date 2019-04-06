
import sys; sys.path.append('../frontend');

import unittest
from frontend import ImageObj

class TestImageObj(unittest.TestCase):
    def test_ImageObj_import(self):
        self.assertTrue(hasattr(ImageObj, "ImageObj"))

    def test_ImageObjConstructorTypeError(self):
        """
        Test that constructor will only accept numpy array
        """
        invalid_types = [set(), dict(), 1, [4,15,6]]
        for inv_arg in invalid_types:
            self.assertRaises(TypeError, ImageObj.ImageObj, inv_arg)

