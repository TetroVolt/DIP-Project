
def resize(src, dsize, fx=None, fy=None, interpolation=0):
    raise NotImplementedError()

def warpAffine(src, M, dsize, dst, flags, borderMode, borderValue):
    output = np.zeros((height, width, 3), dtype = np.uint8)
    input_height, input_width = image.shape[:2]
    for u in range(width):
        for v in range(height):
            x = u * matrix[0, 0] + v * matrix[0,1] + matrix[0,2]
            y = u * matrix[1, 0] + v * matrix[1,1] + matrix[1,2]
            tempx, tempy = int(x), int(y)
            if 0 < tempx < input_width and 0 < tempy < input_height:
                out = image[tempy, tempx]
                output[v, u] = out

    return output
    #raise NotImplementedError()

def getRotationMatrix2D(center, angle, scale):
    width, height = image.shape[:2]
    cx = width / 2
    cy = height / 2

    # calculate rotation matrix
    matrix = rotationMatrix(cx, cy, int(angle), 1)
    #print(matrix)
    cos = np.abs(matrix[0,0])
    sin = np.abs(matrix[0,1])

    # calculate new height and width
    newWidth = int((height * sin) + (width * cos))
    newHeight = int((height * cos) + (width * sin))

    matrix[0, 2] += cx - (newWidth / 2) 
    matrix[1, 2] += cy - (newHeight / 2)
    #raise NotImplementedError()

def getAffineTransform(src, dst):
    raise NotImplementedError()

def getPerspectiveTransform(src, dst, solveMethod = None):
    a = np.zeros((8, 8))
    b = np.zeros((8))
    for i in range(4):
        a[i][0] = a[i + 4][3] = srcPoints[i][0]
        a[i][1] = a[i + 4][4] = srcPoints[i][1]
        a[i][2] = a[i + 4][5] = 1
        a[i][3] = a[i][4] = a[i][5] = 0
        a[i + 4][0] = a[i + 4][1] = a[1 + 4][2] = 0
        a[i][6] = -srcPoints[i][0] * dstPoints[i][0]
        a[i][7] = -srcPoints[i][1] * dstPoints[i][0]
        a[i + 4][6] = -srcPoints[i][0] * dstPoints[i][1]
        a[i + 4][7] = -srcPoints[i][1] * dstPoints[i][1]
        b[i] = dstPoints[i][0]
        b[i + 4] = dstPoints[i][1]

    M = np.linalg.solve(a, b)
    M.resize((9,), refcheck = False)
    M[8] = 1
    return M.reshape((3, 3))
    #raise NotImplementedError()

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
