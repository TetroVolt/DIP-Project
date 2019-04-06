
import numpy as np
import tkinter as TK

class ImageObj:
    """
    this class is for modeling the behavior of an 
    image object for use by the main GUI App to render
    the current image being worked on
    """

    def __init__(self, img_npy:np.array):
        """
            parameters
        """
        if not(isinstance(img_npy, np.array)):
            raise TypeError("img_npy should be of type np.array")

        # TODO check for dtype

        self.__original_img = img_npy
        self.__scale_x, self.__scale_y = 1., 1.
        self.__height, self.__width = self.__original_img.shape

    def scale_XY(self, fx=1., fy=1.):
        """
        meant for scaling for viewing purposes only
        uses nearest neighbor to scale image
        """
        
        pass

    def render(self, canvas: TK.Canvas):
        canvas.create_image()
        pass


    
