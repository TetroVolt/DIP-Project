import numpy as np

def resize(src, dsize, fx=None, fy=None, interpolation=0):
    raise NotImplementedError()

def warp_affine(src, M, dsize, dst, flags, borderMode, borderValue):
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

def get_rotation_matrix_2D(center, angle, scale):
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

def shear_transform(src, dst):
    raise NotImplementedError()

def perspective_transform(src, dst, solveMethod = None):
    raise NotImplementedError()

def warp_perspective(src, M, dsize, dst, flags, borderMode, borderValue):
    raise NotImplementedError()

def get_exports():
    exports = {
        "resize": resize,
        "warp_affine" : warp_affine,
        "get_rotation_matrix_2D": get_rotation_matrix_2D,
        "shear_transform": shear_transform,
        "perspective_transform": perspective_transform,
        "warp_perspective": warp_perspective,
    }

    return exports
