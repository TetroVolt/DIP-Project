
def resize(src, dsize, fx, fy, interpolation):
    pass

def warpAffine(src, M, dsize, dst, flags, borderMode, borderValue):
    pass

def getRotationMatrix2D(center, angle, scale):
    pass

def getAffineTransform(src, dst):
    pass

def getPerspectiveTransform(src, dst, solveMethod):
    pass

def warpPerspective(src, M, dsize, dst, flags, borderMode, borderValue):
    pass

def getExports():
    exports = {
        "resize": resize,
        "warpAffine" : warpAffine,
        "getRotationMatrix2D": getRotationMatrix2D, 
        "getAffineTransform": getAffineTransform, 
        "getPerspectiveTransform": getPerspectiveTransform, 
        "warpPerspective": warpPerspective,
    }

    return exports
