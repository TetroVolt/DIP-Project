import numpy as np

def resize(src, dsize, fx=None, fy=None, interpolation=0):
    raise NotImplementedError()

def warp_affine(src, M, dsize, dst, flags, borderMode, borderValue):
    input_height, input_width = src.shape[:2]
    output = np.zeros((input_height, input_width, 3), dtype = np.uint8)
    
    for u in range(input_width):
        for v in range(input_height):
            x = u * M[0, 0] + v * M[0,1] + M[0,2]
            y = u * M[1, 0] + v * M[1,1] + M[1,2]
            tempx, tempy = int(x), int(y)
            if 0 < tempx < input_width and 0 < tempy < input_height:
                out = image[tempy, tempx]
                output[v, u] = out

    return output

def rotationMatrix(cx, cy, angle, scale):
	m = scale * np.cos(angle * np.pi/180)
	n = scale * (np.sin(angle*np.pi/180))
	u  = (1-m) * cx - n * cy
	v = n * cx + (1-m) * cy
	return np.array([[m, n, u], [-n, m, v]])

def get_rotation_matrix_2D(cx, cy, angle, scale):
    #width, height = image.shape[:2]
    #cx = width / 2
    #cy = height / 2

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

    m = np.zeros((6, 6))
    n = np.zeros((6))    
    for i in range(3):
        j = i * 2
        k = i * 2 + 1
        m[j][0] = m[k][3] = src[i][0]
        m[j][1] = m[k][4] = src[i][1]
        m[i][5] = m[i + 3][5] = m[j][2] = 1
        m[j][3] = m[j][4] = 0
        m[k][0] = m[k][1] = m[k][2] = 0
        n[i*2] = dst[i][0]
        n[i*2+1] = dst[i][1]

    M = np.linalg.solve(m, n)
    return M.reshape(2, 3)
    #raise NotImplementedError()

def perspective_transform(src, dst, solveMethod = None):
    #if srcPoints.shape != (4, 2)   or dstPoints.shape != (4, 2):
    #raise ValueError("There must be four source points and four destination points")

    m = np.zeros((8, 8))
    n = np.zeros((8))
    for i in range(4):
        m[i][0] = m[i + 4][3] = src[i][0]
        m[i][1] = m[i + 4][4] = src[i][1]
        m[i][2] = m[i + 4][5] = 1
        m[i][3] = m[i][4] = a[i][5] = 0
        m[i + 4][0] = m[i + 4][1] = m[1 + 4][2] = 0
        m[i][6] = -src[i][0] * dst[i][0]
        m[i][7] = -src[i][1] * dst[i][0]
        m[i + 4][6] = -src[i][0] * dst[i][1]
        m[i + 4][7] = -src[i][1] * dst[i][1]
        n[i] = dst[i][0]
        n[i + 4] = dst[i][1]

    M = np.linalg.solve(m, n)
    M.resize((9,), refcheck = False)
    M[8] = 1
    return M.reshape((3, 3))
    #raise NotImplementedError()

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
