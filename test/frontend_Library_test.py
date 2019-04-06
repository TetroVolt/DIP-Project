import unittest
import sys; sys.path.append('../frontend');

from frontend import Library
Library = Library.Library

class LibraryTest(unittest.TestCase):

    def testLibraryConstructor(self):
        lib = Library()
    
    def testLibraryConstructorTypeError(self):
        test_args = [1,1., [], (), {}, set(), None]
        for arg in test_args:
            self.assertRaises(TypeError,Library,arg)
    
    def testLibraryConstructorValueError(self):
        alphabet = "abcdefghijklmnopqrstuv"
        from random import choice
        test_args = ["".join([choice(alphabet) for i in range(10)])\
                     for j in range(20)]
        for arg in test_args:
            self.assertRaises(ValueError, Library, arg) 
        