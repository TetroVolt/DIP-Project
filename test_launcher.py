import unittest as test
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

def lib_suite() -> test.TestSuite:
    suite = test.TestSuite()
    loader = test.TestLoader()
    suite.addTest(loader.loadTestsFromTestCase(LibraryTest))
    return suite

def gui_suite() -> test.TestSuite:
    # TODO-Add gui
    suite = test.TestSuite()
    loader = test.TestLoader()
    #suite.addTest(loader.loadTestsFromTestCase(GUITest))
    return suite

if __name__ == "__main__":
    if "--debug" in sys.argv or "-d" in sys.argv:
        sys.argv.pop()
        sys.excepthook = exception_handler

    print("\nRunning Tests...\n")

    # Set a flag so CircleCI knows we failed a test.
    exit_code = 0

    suites = [lib_suite, gui_suite]

    runner = test.TextTestRunner(verbosity=3)
    for suite in suites:
        runner.run(lib_suite())
        if not test.TestResult().wasSuccessful():
            exit_code = 1

    print("\nDone!\n")

    sys.exit(exit_code)

