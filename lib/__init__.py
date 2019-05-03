import numpy as np
# Used to write an image.
import cv2
import math

from typing import Tuple

# Enums for picking the interpolation method.
INTER_NEAREST = 0
INTER_LINEAR = 1
INTER_CUBIC = 2
INTER_LANCZOS4 = 3

# Enums for use in the warpPerspective function to choose
# how the border is dealt with.
BORDER_CONSTANT = 0
BORDER_REPLICATE = 1

def resize(src: np.array, dsize: Tuple[int, int],
           dst=None, fx=None, fy=None, interpolation: int=INTER_LINEAR) -> np.array:
    """
        Wrapper for the appropriate funciton to resample an image based on the interpolation method.

        Args:
            src (:class: array):  The image to be resampled, as a matrix of unsigned intergers.

            dsize (:class: tuple):  A tuple containing the desired size of the new image.

        Kargs:
            scale_x (:class: double):  Amount of scale along the x direction (eg. 0.5, 1.5, 2.5).
                                  Defaults to None.

            scale_y (:class: double):  Amount of scale along the y direction (eg. 0.5, 1.5, 2.5).
                                  Defaults to None.

            interpolation (:class: int):  Method used for interpolation as an integer.
                                  INTER_NEAREST = Nearest Neighbors
                                  INTER_LINEAR = Bilinear
                                  INTER_CUBIC = Bicubic
                                  INTER_LANCZOS4 = Lanczos4
                                  Defaults to bilinear.

        Returns:
            (:class: array):  A resized image based on the interpolation method specified.
    """
    new_rows, new_columns = dsize
    rows, columns = src.shape
    scale_y = float(new_rows) / rows
    scale_x = float(new_columns) / columns
    if interpolation ==  INTER_LINEAR:
        return __bilinear_interpolation(src, (scale_y, scale_x), (new_rows, new_columns))

    elif interpolation == INTER_NEAREST:
        return __nearest_neighbor(src, (scale_y, scale_x), (new_rows, new_columns))

    elif interpolation == INTER_CUBIC:
        return __bicubic_interpolation(src, (scale_y, scale_x), (new_rows, new_columns))

    elif interpolation == INTER_LANCZOS4:
        return __lanczos4_interpolation(src, (scale_y, scale_x), (new_rows, new_columns))


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
            nearest_x = round(column_iter / scale_x)
            nearest_y = round(row_iter / scale_y)

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

def __product_sum(x_arr, x, l):
    coeff = 1
    for index in range(0,4):
        if index != l:
            coeff = coeff * ((x - x_arr[index]) / (x_arr[l] - x_arr[index]))
    return coeff

def __cubic_interpolation(x_vals, intensity_vals, x):
    value = 0
    for index in range(0, 4):
        value = value + (__product_sum(x_vals, x, index) * intensity_vals[index])
    return value

def __bicubic_interpolation(image: np.array, scale: Tuple[float, float], size: Tuple[int, int]) -> np.array:
    """
        Performs a bicubic interpolation rescaling of the desired image.

        Args:
            image (:class: array):  The image to be resampled, as a matrix of unsigned intergers.

            scale (:class: tuple):  Amount of scale (y, x) along each direction (eg. 0.5, 1.5, 2.5).

            size (:class: tuple):  Size of the new image (rows, columns).

        Returns:
            (:class: array):  A resized image based on bicubic interpolation.
    """

    rows, columns = image.shape

    scale_y, scale_x = scale
    new_rows, new_columns = size

    # Create the new image intialized with all zeroes.
    new_image = np.zeros(shape=(new_rows, new_columns), dtype=np.uint8)

    # Initialize the padded version of the original image
    padded_image = image

    # Pad left and right sides of the image
    left_pad = np.append(padded_image[:,0:1],padded_image[:,0:1],axis=1)
    right_pad = np.append(padded_image[:,columns-1:columns],padded_image[:,columns-1:columns],axis=1)
    padded_image = np.append(np.append(left_pad,padded_image, axis=1),right_pad, axis=1)
    # Pad top and bottom of image
    padded_image = np.append(np.append([padded_image[0],padded_image[0]],padded_image, axis=0),[padded_image[rows-1],padded_image[rows-1]], axis=0)
    cv2.imwrite('padded.jpg', padded_image)

    for row_iter in range(0, new_rows):
        for column_iter in range(0, new_columns):

            # This is where the new_image pixel is on the old image.
            # This will likely be a float, in which case we will need to
            # calculate the value.  If it is not, it maps exactly onto an
            # already existing pixel, so use that.
            mapped_x = (column_iter) / scale_x
            mapped_y = (row_iter) / scale_y

            # Get the coordinates for the home pixel,
            # and the rest can be deduced.
            left_x = math.floor(mapped_x)
            top_y = math.floor(mapped_y)
            # Home pixel coordinates relative to padded image
            home_x_p = left_x + 2
            home_y_p = top_y + 2

            # Use the original image value if the mapped values correspond to an exact pixel.
            if mapped_x == left_x and mapped_y == top_y:
                new_pixel = image[top_y, left_x]
                row_counter = math.ceil(scale_x)
                column_counter = math.ceil(scale_y)

            # Interpolate unkown
            else:
                # Matrix of 16 samples for cubic interpolation
                x_coord = [home_x_p-1,home_x_p,home_x_p+1,home_x_p+2]
                y_coord = [home_y_p-1,home_y_p,home_y_p+1,home_y_p+2]
                sample_matrix = padded_image[y_coord[0]:y_coord[3]+1,x_coord[0]:x_coord[3]+1]

                r1 = __cubic_interpolation(x_coord, sample_matrix[0], mapped_x+2)

                r2 = __cubic_interpolation(x_coord, sample_matrix[1], mapped_x+2)

                r3 = __cubic_interpolation(x_coord, sample_matrix[2], mapped_x+2)

                r4 = __cubic_interpolation(x_coord, sample_matrix[3], mapped_x+2)

                new_pixel = __cubic_interpolation(y_coord, [r1,r2,r3,r4], mapped_y+2)

            new_pixel = 0 if new_pixel < 0 else 255 if new_pixel > 255 else new_pixel

            new_image[row_iter, column_iter] = new_pixel

    return new_image

def __lanc_func(x):
    result = 0
    if x <= 4 :
        result = (np.sinc(x)*np.sinc(x/4))
    return result

def __lanczos_weight(x, y):
    weight = 0
    for i in range(-3,5):
        for j in range(-3,5):
            weight = weight + __lanc_func(i-x+int(x))*__lanc_func(j-y+int(y))
    return weight

def __lanczos_filter(x, y, image):
    intensity = 0
    for i in range(-3,5):
        for j in range(-3,5):
            intensity = intensity + image[int(x)+i, int(y)+j]*__lanc_func(i-x+int(x))*__lanc_func(j-y+int(y))
    normalized = intensity/__lanczos_weight(x, y)
    return normalized

def __lanczos4_interpolation(image: np.array, scale: Tuple[float, float], size: Tuple[int, int]) -> np.array:

    rows, columns = image.shape

    scale_y, scale_x = scale
    new_rows, new_columns = size

    # Create the new image intialized with all zeroes.
    new_image = np.zeros(shape=(new_rows, new_columns), dtype=np.uint8)

    # Initialize the padded version of the original image
    padded_image = image

    # Pad left and right sides of the image
    left_pad = np.append(np.append(padded_image[:,0:1],padded_image[:,0:1],axis=1),np.append(padded_image[:,0:1],padded_image[:,0:1],axis=1),axis=1)
    right_pad = np.append(np.append(padded_image[:,columns-1:columns],padded_image[:,columns-1:columns],axis=1),np.append(padded_image[:,columns-1:columns],padded_image[:,columns-1:columns],axis=1),axis=1)
    padded_image = np.append(np.append(left_pad,padded_image, axis=1),right_pad, axis=1)
    # Pad top and bottom of image
    top_pad = np.array([padded_image[0], padded_image[0], padded_image[0], padded_image[0]])
    bot_pad = np.array([padded_image[rows-1], padded_image[rows-1], padded_image[rows-1], padded_image[rows-1]])
    padded_image = np.append(np.append(top_pad,padded_image, axis=0),bot_pad, axis=0)

    for row_iter in range(0, new_rows):
        for column_iter in range(0, new_columns):

            # This is where the new_image pixel is on the old image.
            # This will likely be a float, in which case we will need to
            # calculate the value.  If it is not, it maps exactly onto an
            # already existing pixel, so use that.
            mapped_x = (column_iter) / scale_x
            mapped_y = (row_iter) / scale_y

            # Get the coordinates for the home pixel,
            # and the rest can be deduced.
            left_x = math.floor(mapped_x)
            top_y = math.floor(mapped_y)
            # Home pixel coordinates relative to padded image
            home_x_p = left_x + 4
            home_y_p = top_y + 4

            new_pixel = __lanczos_filter(mapped_y + 4, mapped_x + 4, padded_image)
            if new_pixel < 0 :
                new_pixel = 0
            elif new_pixel > 255 :
                new_pixel = 255

            new_image[row_iter,column_iter] = new_pixel

    return new_image

def __warp_cubic(x_array, y_coord, image, x):

    adjusted_y = y_coord

    if adjusted_y < 0:
        adjusted_y = 0
    elif adjusted_y >= image.shape[0]:
        adjusted_y = image.shape[0] - 1

    intensity_array = np.zeros(shape=4)
    for i in range(0,4):
        temp_x = x_array[i]
        if x_array[i] < 0:
            temp_x = 0
        elif x_array[i] >= image.shape[1]:
            temp_x = image.shape[1] - 1
        intensity_array[i] = image[adjusted_y,temp_x]

    return __cubic_interpolation(x_array, intensity_array, x)

def warpAffine(image: np.array, transform: np.array, size: Tuple[int, int], interpolation: int) -> np.array:
    """
        Apply the affine transformation to an image given a 2x3 transformation matrix.
        The basic idea is to apply a transform matrix to each pixel, which will offset them
        by the desired amount.  You multiply the pixel (x, y) vector by 4 elements of the matrix,
        and then add an offset vector to that.  This will result in different pixels being offset by
        different amounts.  The formula for it is below in the code.

        Args:
            image (:class: array):  The image to perform a transformation on.

            transform (:class: array):  A 2x3 transformation matrix.  This is used to remap each pixel.

            size (:class: tuple):  The size of the new image (rows, columns).

            interpolation (:class: int):  The interpolation method to use for mapping pixels.

        Returns:
            (:class: array):  The transformed image of the specified image.
    """
    if transform.shape != (2, 3):
        raise ValueError("Transform Matrix is the incorrect size.")
    rows, columns = image.shape
    output = np.zeros((rows, columns), dtype = np.uint8)

    for row in range(0, rows):
        for column in range(0, columns):
            # Formula used:
            # [x'] = [a b][x]+[t1] Where [a b t1] is the transform matrix.
            # [y']   [c d][y] [t2]       [c d t2]

            mapped_x = transform[0, 0] * column + transform[0,1] * row + transform[0,2]
            mapped_y = transform[1, 0] * column + transform[1,1] * row + transform[1,2]
            int_x = int(mapped_x)
            int_y = int(mapped_y)

            if row == 511 and column == 511:
                import pdb;pdb.set_trace()

            # Ensure we don't go out of bounds of the image.
            # If we go outside the bounds, the result will just be 0, as the image
            # was initialized with all zeroes.
            if 0 < mapped_x < columns and 0 < mapped_y < rows:

                # If our mapped values correspond to actual pixels, use those.
                if mapped_x.is_integer() and mapped_y.is_integer():
                    output[row, column] = image[int_y, int_x]

                # If we get a mapped value that is not an exact coordinate, we need to interpolate.
                elif interpolation == INTER_NEAREST:
                    # This one's easy, just find the closest points.
                    output[row, column] = image[int(round(mapped_y)), int(round(mapped_x))]

                elif interpolation == INTER_LINEAR:
                    left_x, right_x = int_x, int_x+1
                    top_y, bottom_y = int_y, int_y+1

                    # Get our two imaginary points for X value.
                    r1 = __interpolate(right_x, mapped_x, left_x,
                                       image[top_y-1, left_x-1], image[top_y-1, left_x])
                    r2 = __interpolate(right_x, mapped_x, left_x,
                                       image[top_y, left_x-1], image[top_y, left_x])

                    # Interpolate the Y value and assign it to the image.
                    output[row, column] = __interpolate(bottom_y, mapped_y, top_y, r1, r2)

                elif interpolation == INTER_CUBIC:
                    # Matrix of 16 samples for cubic interpolation.
                    x_coord = [int_x-1,int_x,int_x+1,int_x+2]
                    y_coord = [int_y-1,int_y,int_y+1,int_y+2]

                    r1 = __warp_cubic(x_coord, y_coord[0], image, mapped_x)

                    r2 = __warp_cubic(x_coord, y_coord[1], image, mapped_x)

                    r3 = __warp_cubic(x_coord, y_coord[2], image, mapped_x)

                    r4 = __warp_cubic(x_coord, y_coord[3], image, mapped_x)

                    new_pixel = __cubic_interpolation(y_coord, [r1,r2,r3,r4], mapped_y)

                    new_pixel = 0 if new_pixel < 0 else 255 if new_pixel > 255 else new_pixel

                    output[row, column] = new_pixel

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
    for index in range(4):
        m[index][0] = m[index + 4][3] = src[index][0]
        m[index][1] = m[index + 4][4] = src[index][1]
        m[index][2] = m[index + 4][5] = 1
        m[index][3] = m[index][4] = m[index][5] = 0
        m[index + 4][0] = m[index + 4][1] = m[1 + 4][2] = 0
        m[index][6] = -src[index][0] * dst[index][0]
        m[index][7] = -src[index][1] * dst[index][0]
        m[index + 4][6] = -src[index][0] * dst[index][1]
        m[index + 4][7] = -src[index][1] * dst[index][1]
        n[index] = dst[index][0]
        n[index + 4] = dst[index][1]

    transform_matrix = np.linalg.solve(m, n)
    transform_matrix.resize((9,), refcheck = False)
    transform_matrix[8] = 1
    # Reshape it to be the needed 3,3 matrix.
    return transform_matrix.reshape((3, 3))

def warpPerspective(image, transform, dsize,
                    flags=(INTER_LINEAR, 0), borderMode=BORDER_CONSTANT, borderValue=0):
    """
        Creates a scaled and/or rotated version of an image using a set of 4 coordinate source points and
        4 coordinate destination points.  Straight lines remain straight during this process.  This takes
        in an already generated transformation matrix to offset each point by.

        Args:
            image (:class: np.array):  The source image to transform

            transform (:class: np.array):  A 3X3 transformation array obtained via "getPerspectiveTransform".

            dsize (:class: tuple):  The size of the new image.
        KArgs:
            flags (:class: tuple):  A tuple containing the desired interpolation method, and whether or not
                to do an inverse transform.  Defaults to using binlinear and non-inverse.

            borderMode (:class: int):  The desired mode for dealing with the borders if an image is out of
                bounds.  Defaults to using a single constant value.

            borderValue (:class: int):  If BORDER_CONSTANT is used, this is the value to be used around the
                border of the image.  Defaults to 0 (black).
    """
    if transform.shape != (3, 3):
        raise ValueError("Transform Matrix must be 3X3")
    rows, columns = image.shape
    new_rows, new_columns = dsize
    # Initialize the image with the desired border value.  If we get a mapped value outside the image,
    # we would use this instead.  If BORDER_REPLICATE is desired, handle that below.
    output = np.full((new_rows, new_columns), borderValue, dtype = np.uint8)

    transform = np.linalg.inv(transform)

    for row in range(0, new_rows):
        for column in range(0, new_columns):
            # M in this case is our transform_matrix.
            # x and y is the coordinates of the new image.
            # The value of the new image at 0, 0 would be the mapped coordinates this formula determines,
            # and the pixel value would be obtained from the old image.
            # [new x] = [(M11 * x + M12 * y + M13)/(M31 * x + M32 * y + M33)]
            # [new y]   [(M21 * x + M22 * y + M23)/(M31 * x + M32 * y + M33)]
            mapped_x = (transform[0, 0] * column + transform[0, 1] * row + transform[0, 2]) / \
                       (transform[2, 0] * column + transform[2, 1] * row + transform[2, 2])
            mapped_y = (transform[1, 0] * column + transform[1, 1] * row + transform[1, 2]) / \
                       (transform[2, 0] * column + transform[2, 1] * row + transform[2, 2])
            int_x = int(mapped_x)
            int_y = int(mapped_y)

            # Ensure we don't go out of bounds of the image.
            # If we go outside the bounds, the result will just be 0, as the image
            # was initialized with a border value.
            if mapped_x > 0 and mapped_y > 0 and mapped_x < columns and mapped_y < rows:
                # If our mapped values correspond to actual pixels, use those.
                if mapped_x.is_integer() and mapped_y.is_integer():
                    output[row, column] = image[int_x, int_y]

                # If we get a mapped value that is not an exact coordinate, we need to interpolate.
                elif flags[0] == INTER_LINEAR:
                    # This one's easy, just find the closest points.
                    output[row, column] = image[int(round(mapped_y)), int(round(mapped_x))]

                elif flags[0] == INTER_LINEAR:
                    left_x, right_x = int_x, int_x+1
                    top_y, bottom_y = int_y, int_y+1

                    # Get our two imaginary points for X value.
                    r1 = __interpolate(right_x, mapped_x, left_x,
                                       image[top_y-1, left_x-1], image[top_y-1, left_x])
                    r2 = __interpolate(right_x, mapped_x, left_x,
                                       image[top_y, left_x-1], image[top_y, left_x])

                    # Interpolate the Y value and assign it to the image.
                    output[row, column] = __interpolate(bottom_y, mapped_y, top_y, r1, r2)

                elif flags[0] == INTER_CUBIC:
                    # Matrix of 16 samples for cubic interpolation.
                    x_coord = [int_x-1,int_x,int_x+1,int_x+2]
                    y_coord = [int_y-1,int_y,int_y+1,int_y+2]

                    r1 = __warp_cubic(x_coord, y_coord[0], image, mapped_x)

                    r2 = __warp_cubic(x_coord, y_coord[1], image, mapped_x)

                    r3 = __warp_cubic(x_coord, y_coord[2], image, mapped_x)

                    r4 = __warp_cubic(x_coord, y_coord[3], image, mapped_x)

                    new_pixel = __cubic_interpolation(y_coord, [r1,r2,r3,r4], mapped_y)

                    new_pixel = 0 if new_pixel < 0 else 255 if new_pixel > 255 else new_pixel

                    output[row, column] = new_pixel

    return output

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

