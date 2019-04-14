import numpy as np
import math
import math

INTER_NEAREST = 0
INTER_LINEAR = 1

def resize(image, output_size, dst=None, fx=None, fy=None, interpolation=INTER_LINEAR):
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
        return bilinear_interpolation(image, (scale_y, scale_x), (new_rows, new_columns))

    elif interpolation == INTER_NEIGHBOR:
        return nearest_neighbor(image, (scale_y, scale_x), (new_rows, new_columns))

def nearest_neighbor(image, scale, size):
    """
        Performs a neartest neighbor scaling of the desired image.

        Args:
            image (:class: array):  The image to be resampled, as a matrix of unsigned intergers.

            scale_x (:class: double):  Amount of scale along the x direction (eg. 0.5, 1.5, 2.5).

            scale_y (:class: double):  Amount of scale along the y direction (eg. 0.5, 1.5, 2.5).

        Returns:
            (:class: array):  A resized image based on the interpolation method specified.
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

def interpolate(right, mapped_val, left, pixel1, pixel2):
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
    """
    return ((right - mapped_val) / (right - left)) * pixel1 + \
           ((mapped_val - left) / (right - left)) * pixel2

def bilinear_interpolation(image, scale, size):
    """
        Performs a bilinear interpolation rescaling of the desired image.
        Formula obtained from Wikipedia here: https://en.wikipedia.org/wiki/Bilinear_interpolation

        Args:
            image (:class: array):  The image to be resampled, as a matrix of unsigned intergers.

            scale_x (:class: double):  Amount of scale along the x direction (eg. 0.5, 1.5, 2.5).

            scale_y (:class: double):  Amount of scale along the y direction (eg. 0.5, 1.5, 2.5).

        Returns:
            (:class: array):  A resized image based on the interpolation method specified.
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
                    r1 = interpolate(right_x, mapped_x, left_x,
                                     image[top_y, left_x-1], image[top_y, left_x])

                    r2 = interpolate(right_x, mapped_x, left_x,
                                     image[bottom_y, left_x-1], image[bottom_y, left_x])

                # We are at the bottom edge of the original image.
                elif left_x < columns-1 and top_y == rows-1:
                    # r2 is out of bounds for the next y, so just use the values from the row above for
                    # r1, and r2 will be the row that are currently on.  The four closest real pixels
                    # would still be these values.
                    r1 = interpolate(right_x, mapped_x, left_x,
                                     image[top_y-1, left_x], image[top_y-1, right_x])

                    r2 = interpolate(right_x, mapped_x, left_x,
                                     image[top_y, left_x], image[top_y, right_x])

                # We are at the bottom right corner of the original image.
                elif left_x == columns-1 and top_y == rows-1:
                    # Use the current nearest pixel as the bottom right corner of our interpolation.
                    r1 = interpolate(right_x, mapped_x, left_x,
                                     image[top_y-1, left_x-1], image[top_y-1, left_x])

                    r2 = interpolate(right_x, mapped_x, left_x,
                                     image[top_y, left_x-1], image[top_y, left_x])
                # It's not a corner or an edge, so calculate using four known points.
                else:
                    r1 = interpolate(right_x, mapped_x, left_x,
                                     image[top_y, left_x], image[top_y, right_x])

                    r2 = interpolate(right_x, mapped_x, left_x,
                                     image[bottom_y, left_x], image[bottom_y, right_x])

                # This is our P value.
                new_pixel = interpolate(bottom_y, mapped_y, top_y, r1, r2)

            new_image[row_iter, column_iter] = new_pixel

    return new_image


def warpAffine(src, M, dsize, dst, flags, borderMode, borderValue):
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

def rotationMatrix2D(center: tuple, angle: float, scale: float) -> np.array:
    """
    parameters:
        center: tuple of 2 numbers representing center of rotation
        angle: specifies amount of counter-clockwise rotation in degrees
        scale: specifies scaling amount to increase magnitude of unit vectors by
    """
    angle = math.radians(angle)
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
    #raise NotImplementedError()

def perspectiveTransform(src, dst, solveMethod = None):
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

def warpPerspective(src, M, dsize, dst, flags, borderMode, borderValue):
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
