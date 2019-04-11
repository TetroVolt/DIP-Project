import unittest
import sys

from lib.test_library import LibraryTest

def exception_handler(exception_type, exception, traceback,
                      debug_hook=sys.excepthook):
    """
        Overwrite the default exception handler to allow for post mortem debugging.
    """
    import pdb
    debug_hook(exception_type, exception, traceback)
    pdb.post_mortem(traceback)

def lib_suite():
    suite = unittest.TestSuite()
    suite.addTest(LibraryTest('test_resize'))
    return suite

def gui_suite():
    # TODO
    suite = unittest.TestSuite()
    return suite

if __name__ == "__main__":
    if "--debug" in sys.argv or "-d" in sys.argv:
        sys.argv.pop()
        sys.excepthook = exception_handler
    runner = unittest.TextTestRunner()
    runner.run(lib_suite())
    runner = unittest.TextTestRunner()
    runner.run(gui_suite())
    unittest.main(verbosity=2)

