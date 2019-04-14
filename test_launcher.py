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

    # Set a flag so CircleCI knows we failed a test.
    failed = 1

    # TODO: Once lib is working, add it.
    suites = [gui_suite]

    runner = unittest.TextTestRunner(verbosity=3)
    for suite in suites:
        runner.run(suite())
        if not unittest.TestResult().wasSuccessful():
            failed = 1

    print("\nDone!\n")

    sys.exit(failed)

