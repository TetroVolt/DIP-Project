
import re

class Library:
    """
    Library Dependency Injection
    """
    def __init__(self, source:str="opencv"):
        if source == "opencv":
            import cv2
            self.lib = cv2
        elif source == "library":
            import sys
            sys.path.append('../')
            import lib
            self.lib = lib
        elif isinstance(source, str):
            raise ValueError("source argument must either be 'opencv' or 'library'")
        else:
            raise TypeError("source argument must be of type str")    

    def __getattr__(self, attr):
        return self.lib.__getattribute__(attr)


TransformFunctions = Library('opencv')
