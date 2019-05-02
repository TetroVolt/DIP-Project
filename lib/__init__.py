import numpy as np
import math

from typing import Tuple

INTER_NEAREST = 0
INTER_LINEAR = 1

def resize(image: np.array, output_size: Tuple[int, int],
           dst=None, fx=None, fy=None, interpolation: int=INTER_LINEAR) -> np.array:
    """
        Wrapper for the appropriate funciton to resample an image based on the interpolation method.

        Args:
            image (:class: array):  The image to be resampled, as a matrix of unsigned intergers.

            output_size (:class: tuple):  A tuple containing the desired size of the new image.

        Kargs:
            scale_x (:class: double):  Amount of scale along the x direction (eg. 0.5, 1.5, 2.5).
                                  Defaults to None.

            scale_y (:class: double):  Amount of scale along the y direction (eg. 0.5, 1.5, 2.5).
                                  Defaults to None.

            interpolation (:class: int):  Method used for interpolation as an integer (either bilinear or
                                          nearest_neighbor).  Defaults to bilinear.

        Returns:
            (:class: array):  A resized image based on the interpolation method specified.
    """
    new_rows, new_columns = output_size
    rows, columns = image.shape
    scale_y = float(new_rows) / rows
    scale_x = float(new_columns) / columns
    if interpolation ==  INTER_LINEAR:
        return __bilinear_interpolation(image, (scale_y, scale_x), (new_rows, new_columns))

    elif interpolation == INTER_NEIGHBOR:
        return __nearest_neighbor(image, (scale_y, scale_x), (new_rows, new_columns))

def __nearest_neighbor(image: np.array, scale: Tuple[float, float], size: Tuple[int, int]) -> np.array:
    """
        Performs a neartest neighbor scaling of the desired image.

        Args:
            image (:class: array):  The image to be resampled, as a matrix of unsigned intergers.

            scale (:class: tuple):  Amount of scale (y, x) along each direction (eg. 0.5, 1.5, 2.5).

            size (:class: tuple):  Size of the new image (rows, columns).

        Returns:
            (:class: array):  A resized image based on nearest neighbor interpolation..
    """
   # Get the dimensions of the current image.
    rows = len(image)
    columns = len(image[0])

    scale_y, scale_x = scale
    new_rows, new_columns = size

    # Create the new image initialized with all zeroes.
    new_image = np.zeros((new_rows, new_columns), dtype=np.uint8)

    for row_iter in range(0, new_rows):
        for column_iter in range(0, new_columns):
            # Calculate where the closest point is on the old image based on the scale factor.
            # Then round that as appropriate.
            nearest_x = integer_round(column_iter / scale_x)
            nearest_y = integer_round(row_iter / scale_y)

            # Santize the x value to be within bounds.
            # If the actual value is 255.5 then it will round out of bounds.
            if nearest_x >= columns:
                nearest_x = columns-1

            if nearest_y >= rows:
                nearest_y = rows-1

            # Assign the image to the old image's position.
            new_image[row_iter, column_iter] = image[nearest_y, nearest_x]
    return new_image

def __interpolate(right: int, mapped_val: float, left: int, pixel1: int, pixel2: int) -> float:
    """
        Performs the interpolation mathematics.

        Args:
            right (:class: int):  The x^2 value of the source image.

            mapped_val (:class: float):  The mapped value of where the new image
                       points to the source image. This will be a point between two
                       existing source image pixels.

           left (:class: int):  The x^1 value of the source image.

           pixel1 (:class: int):  The source image pixel value at x^2.

           pixel2 (:class: int):  The source image pixel value at x^1.

        Returns:
            (:class: float):  The result.
    """
    return ((right - mapped_val) / (right - left)) * pixel1 + \
           ((mapped_val - left) / (right - left)) * pixel2

def __bilinear_interpolation(image: np.array, scale: Tuple[float, float], size: Tuple[int, int]) -> np.array:
    """
        Performs a bilinear interpolation rescaling of the desired image.
        Formula obtained from Wikipedia here: https://en.wikipedia.org/wiki/Bilinear_interpolation

        Args:
            image (:class: array):  The image to be resampled, as a matrix of unsigned intergers.

            scale (:class: tuple):  Amount of scale (y, x) along each direction (eg. 0.5, 1.5, 2.5).

            size (:class: tuple):  Size of the new image (rows, columns).

        Returns:
            (:class: array):  A resized image based on bilinear interpolation.
    """
    # Get the dimensions of the current image.
    rows, columns = image.shape

    scale_y, scale_x = scale
    new_rows, new_columns = size

    # Create the new image intialized with all zeroes.
    new_image = np.zeros(shape=(new_rows, new_columns), dtype=np.uint8)

    for row_iter in range(0, new_rows):
        for column_iter in range(0, new_columns):

            # This is where the new_image pixel is on the old image.
            # This will likely be a float, in which case we will need to
            # calculate the value.  If it is not, it maps exactly onto an
            # already existing pixel, so use that.
            mapped_x = (column_iter +0.5)/ scale_x
            mapped_y = (row_iter+0.5) / scale_y

            # Get the coordinates for the top left most corner of the interpolation square,
            # and the rest can be deduced.
            left_x = math.floor(mapped_x)
            top_y = math.floor(mapped_y)

            # Use the original image value if the mapped values correspond to an exact pixel.
            if mapped_x == left_x and mapped_y == top_y:
                new_pixel = image[top_y, left_x]
                row_counter = ceil(scale_x)
                column_counter = ceil(scale_y)

            # Calculate the imaginary value.
            else:
                # This is our x^2.
                # x^1 is column_iter.
                right_x = left_x + 1

                # This is our y^2.
                # y^1 is row_iter.
                bottom_y = top_y + 1

                # We are at the right edge of the original image.
                if left_x == columns-1 and top_y < rows-1:
                    # r1 and r2 is out of bounds for the next x, so just use the values from the previous
                    # column and the column we are currently on.  The four closest real pixels
                    # would still be these values.
                    r1 = __interpolate(right_x, mapped_x, left_x,
                                       image[top_y, left_x-1], image[top_y, left_x])

                    r2 = __interpolate(right_x, mapped_x, left_x,
                                       image[bottom_y, left_x-1], image[bottom_y, left_x])

                # We are at the bottom edge of the original image.
                elif left_x < columns-1 and top_y == rows-1:
                    # r2 is out of bounds for the next y, so just use the values from the row above for
                    # r1, and r2 will be the row that are currently on.  The four closest real pixels
                    # would still be these values.
                    r1 = __interpolate(right_x, mapped_x, left_x,
                                       image[top_y-1, left_x], image[top_y-1, right_x])

                    r2 = __interpolate(right_x, mapped_x, left_x,
                                       image[top_y, left_x], image[top_y, right_x])

                # We are at the bottom right corner of the original image.
                elif left_x == columns-1 and top_y == rows-1:
                    # Use the current nearest pixel as the bottom right corner of our interpolation.
                    r1 = __interpolate(right_x, mapped_x, left_x,
                                       image[top_y-1, left_x-1], image[top_y-1, left_x])

                    r2 = __interpolate(right_x, mapped_x, left_x,
                                       image[top_y, left_x-1], image[top_y, left_x])
                # It's not a corner or an edge, so calculate using four known points.
                else:
                    r1 = __interpolate(right_x, mapped_x, left_x,
                                       image[top_y, left_x], image[top_y, right_x])

                    r2 = __interpolate(right_x, mapped_x, left_x,
                                       image[bottom_y, left_x], image[bottom_y, right_x])

                # This is our P value.
                new_pixel = __interpolate(bottom_y, mapped_y, top_y, r1, r2)

            new_image[row_iter, column_iter] = new_pixel

    return new_image

def warpAffine(image: np.array, transform: np.array, size: Tuple[int, int]) -> np.array:
    """
        Apply the affine transformation to an image given a 2x3 transformation matrix.

        Args:
            image (:class: array):  The image to perform a transformation on.

            transform (:class: array):  A 2x3 transformation matrix.  This is used to remap each pixel.

            size (:class: tuple):  The size of the new image (rows, columns).

        Returns:
            (:class: array):  The transformed image of the specified image.
    """
    rows, columns = image.shape
    output = np.zeros((rows, columns), dtype = np.uint8)

    for row in range(0, rows):
        for column in range(0, columns):
            # [x'] = [a b][x]+[t1] Where [a b t1] is the transform matrix.
            # [y']   [c d][y] [t2]       [c d t2]
            new_x = int(row * transform[0, 0] + column * transform[0,1] + transform[0,2])
            new_y = int(row * transform[1, 0] + column * transform[1,1] + transform[1,2])
            # Ensure we don't go out of bounds of the image.
            if 0 < new_x < columns and 0 < new_y < rows:
                output[row, column] = image[new_y, new_x]

    return output

def getRotationMatrix2D(center: Tuple[float, float], angle: float, scale: float, radians: bool=False) -> np.array:
    """
        Gets a two dimensional rotation matrix with adjustable center of rotation.
        Used as an input for an affine transform.  This will not transform anything by itself.

        Args:
            center (:class: tuple): Tuple of two floats representing (y, x) coordinate of the center.

            angle (:class: float): Specifies the amount of counter-clockwise rotation in degrees.
                If radians=True, this is expected to be in radians.

            scale (:class: float): Specifies how much to increases the magnitude of the unit vectors.

        Kargs:
            radians (:class: bool):  If true, then the angle paramater is expected to be in radians.
                Otherwise it is expected to be in degrees.  Defaults to false.
    """
    angle = math.radians(angle) if not radians else angle
    center_x, center_y = center

    alpha = scale * math.cos(angle)
    beta  = scale * math.sin(angle)
    shift_x = (1-alpha) * center_x - beta * center_y
    shift_y = beta * center_x + (1-alpha) * center_y

    return np.array([[alpha, beta , shift_x],
                     [-beta, alpha, shift_y]], dtype=np.float)

def shearTransform(src, dst):
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

def getPerspectiveTransform(src: np.array, dst: np.array, solveMethod=None) -> np.array:
    """
        The calculates the 3x3 transformation matrix to be passed into "warpPerspective".

        Args:
            src (:class: np.array):  A set of 4 source coordinates to start the warp from.

            dst (:class: np.array): A set of 4 destination coordinates to scale everything to.

        Returns:
            (:class: np.array):  A 3x3 transformation matrix.
    """
    # Sanity check the input.
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("There must be four source points and four destination points.")

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

def fisheye(image: np.array) -> np.array:
    """
        Create a "fisheye" distortion on the provided image.

        Args:
            image (:class: np.array):  The image to distort.

        Returns:
            (:class: np.array):  The distorted image.
    """
    #Creates an array of index positions for the image.
    x = np.size(image, 0)
    y = np.size(image, 1)

    r0 = np.arange(y)
    r1 = np.arange(x)

    out = np.empty((y, x, 2), dtype=float)

    out[:, :, 0] = r0[:, None]
    out[:, :, 1] = r1

    xy = out
    #Reshapes array into a a form of x, y coordinate pairs
    xy = xy.reshape(-1, 2)
    #creates final array
    final = np.zeros((x, y))
    #Finds Center of the image
    center = np.mean(xy, axis=0)
    #create 2 arrays of distances from the cinter for x and y coordinates
    xc, yc = (xy - center).T

    # Polar coordinates
    r = np.sqrt(xc**2 + yc**2)
    #creates theta array
    theta = np.arctan2(yc, xc)
    #changes the radius to the shorter radius for distortion
    rd = r * 0.8999999999999
    #normalizes the radius
    normR = (rd - min(rd)) / (max(rd)-min(rd))

    #generates new distorted radial distances stretching the image
    r = rd*(1 + (-.7) * normR) +.5
    #generates the mask that will be used to map the original image to the new image
    mask = np.column_stack((r * np.cos(theta), r * np.sin(theta))) + center
    #Nearest Neighbor Mapping
    height, width = mask.shape
    for i in range(0, height):
        nX = math.floor(mask[i][0] + .05)
        nY = math.floor(mask[i][1] + .05)
        x = xy[i][0] + .05
        y = xy[i][1] + .05

        final[nY][nX] = image[math.floor(y)][math.floor(x)]
    #return final image
    return final

def warpPerspective(src, M, dsize, dst, flags, borderMode, borderValue):
    """
        Creates a scaled and/or rotated version of an image using a set of 4 coordinate source points and
        4 coordinate destination points.  Straight lines remain straight during this process.
    """
    raise NotImplementedError()

def get_exports() -> dict:
    """
        This returns all the module functions as a dictionary map.
    """
    return {
        "resize": resize,
        "warpAffine" : warpAffine,
        "getRotationMatrix2D": getRotationMatrix2D,
        "shearTransform": shearTransform,
        "getPerspectiveTransform": getPerspectiveTransform,
        "warpPerspective": warpPerspective,
        "fisheye": fisheye
    }

