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
    loader = unittest.TestLoader()
    suite.addTest(loader.loadTestsFromTestCase(LibraryTest))
    return suite

def gui_suite():
    # TODO-Add gui
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    #suite.addTest(loader.loadTestsFromTestCase(GUITest))
    return suite

if __name__ == "__main__":
    if "--debug" in sys.argv or "-d" in sys.argv:
        sys.argv.pop()
        sys.excepthook = exception_handler

    print("\nRunning Tests...\n")

    runner = unittest.TextTestRunner(verbosity=3)
    runner.run(lib_suite())
    runner.run(gui_suite())

    print("\nDone!\n")

