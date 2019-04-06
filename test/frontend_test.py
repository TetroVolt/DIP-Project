
import sys; sys.path.append('..');

import unittest
from frontend import frontend

class TestFrontEnd(unittest.TestCase):
    def test_frontend(self):
        self.assertTrue(True, msg="Is not true.")
    
    def test_front_end_import(self):
        print(dir(frontend))

if __name__=='__main__':
    unittest.main()

