
import sys; sys.path.append('../frontend');

import unittest
from frontend import frontend

class TestFrontEnd(unittest.TestCase):
    def test_front_end_import(self):
        self.assertTrue(hasattr(frontend, "App"))
    


if __name__=='__main__':
    unittest.main()

