
import re

class SingletonDecorator:
    def __init__(self, klass):
        self.klass = klass
        self.instance = None
    
    def __call__(self, *args, **kwargs):
        if self.instance is None:
            self.instance = self.klass(*args, **kwargs)
        return self.instance

class Library:
    """
    Library Dependency Injection
    """
    def __init__(self, source:str="opencv"):
        self.source = source

        import sys
        sys.path.append('../')
        import lib
        import cv2
        self.lib = lib
        self.cv2 = cv2

        if source == "opencv":
            self.libsource = cv2
        elif source == "library":
            self.libsource = lib
        elif isinstance(source, str):
            raise ValueError("source argument must either be 'opencv' or 'library'")
        else:
            raise TypeError("source argument must be of type str")    

    def __getattr__(self, attr):
        if attr == 'fisheye':
            return self.lib.fisheye
        else:
            return self.libsource.__getattribute__(attr)

Library = SingletonDecorator(Library)
lib = Library('opencv')
#lib = Library('library')
