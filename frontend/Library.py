
class Library:
    """
    Library Dependency Injection
    """
    def __init__(self, source:str="opencv"):
        if source == "opencv":
            self.__use_cv2()
        elif source == "library":
            self.__use_library()
        elif isinstance(source, str):
            raise ValueError("source argument must either be 'opencv' or 'library'")
        else:
            raise TypeError("source argument must be of type str")    

    def __use_cv2(self):
        import cv2 as cv
        self.resize = cv.resize
        self.warpAffine = cv.warpAffine
        self.getRotationMatrix = cv.getRotationMatrix2D
        self.getAffineTransform = cv.getAffineTransform
        self.getPerspectiveTransform = cv.getPerspectiveTransform
        self.warpPerspective = cv.warpPerspective

