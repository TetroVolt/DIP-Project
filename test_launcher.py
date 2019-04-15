import unittest
import sys
import argparse
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', help="set debug on (PDB)")
    parser.add_argument('-g', '--gui', action='store_true', help='test gui only')
    parser.add_argument('-l', '--lib', action='store_true', help='test library only')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        sys.excepthook = exception_handler

    print("\nRunning Tests...\n")
        
    if args.gui:
        suites = [gui_suite]        
    elif args.lib:
        suites = [lib_suite]
    else:
        suites = [gui_suite, lib_suite]
    
    exit_code = 0
    runner = unittest.TextTestRunner(verbosity=3)
    for suite in suites:
        runner.run(suite())
        if not unittest.TestResult().wasSuccessful():
            exit_code += 1

    print("\nDone!\n")
        
    sys.exit(exit_code)