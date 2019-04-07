
def resize(src, dsize, fx=None, fy=None, interpolation=0):
    raise NotImplementedError()

def warpAffine(src, M, dsize, dst, flags, borderMode, borderValue):
    raise NotImplementedError()

def getRotationMatrix2D(center, angle, scale):
    raise NotImplementedError()

def getAffineTransform(src, dst):
    raise NotImplementedError()

def getPerspectiveTransform(src, dst, solveMethod):
    raise NotImplementedError()

def warpPerspective(src, M, dsize, dst, flags, borderMode, borderValue):
    raise NotImplementedError()

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
