import unittest
import sys; sys.path.append('../frontend');

import numpy as np
from frontend import Library
Library = Library.Library

class LibraryTest(unittest.TestCase):

    def testLibraryConstructor(self):
        lib = Library()
    
    def testLibraryConstructorTypeError(self):
        test_args = [1, 0.0,  float, [], (), dict, set(), None]
        for arg in test_args:
            self.assertRaises(TypeError,Library,arg)
    
    def testLibraryConstructorValueError(self):
        alphabet = "abcdefghijklmnopqrstuv"
        from random import choice
        test_args = ["".join([choice(alphabet) for i in range(10)])\
                     for j in range(20)]
        for arg in test_args:
            self.assertRaises(ValueError, Library, arg) 
        
    def testLibraryAttributesOpenCV(self):
        attributes = [
            "resize",
            "warpAffine",
            "getRotationMatrix2D",
            "getAffineTransform", 
            "getPerspectiveTransform", 
            "warpPerspective",
            ]
        lib_opencv = Library("opencv")
        for attr in attributes:
            self.assertTrue(hasattr(lib_opencv, attr), "source 'opencv' has no attr: " + attr)

    def testLibraryAttributesLibrary(self):
        attributes = [
            "resize",
            "warpAffine",
            "getRotationMatrix2D",
            "getAffineTransform",
            "getPerspectiveTransform",
            "warpPerspective",
            ]
        lib_opencv = Library("library")
        for attr in attributes:
            self.assertTrue(hasattr(lib_opencv, attr), "source 'library' has no attr: " + attr)

    def testLibrary_resize(self):
        libcv = Library('opencv')
        liblib = Library('library')
        self.assertTrue(hasattr(libcv, 'resize'))
        self.assertTrue(hasattr(liblib, 'resize'))
        
        args = [np.random.rand(10,10), (20,20)]
        self.assertEqual(libcv.resize(*args), liblib.resize(*args), "Resize function not equal.")

    def testLibrary_warpAffine(self):
        pass
    def testLibrary_getRotationMatrix2D(self):
        pass
    def testLibrary_getAffineTransform(self):
        pass
    def testLibrary_getPerspectiveTransform(self):
        pass
    def testLibrary_warpPerspective(self):
        pass
