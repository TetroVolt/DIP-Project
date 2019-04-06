def resize():
    pass
def warpAffine():
    pass
def getRotationMatrix2D():
    pass
def getAffineTransform():
    pass
def getPerspectiveTransform():
    pass
def warpPerspective():
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
