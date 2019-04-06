
import numpy as np

class ImageObj:
    """
    this class is for modeling the behavior of an 
    image object for use by the main GUI App to render
    the current image being worked on
    """

    def __init__(self, img_npy: np.array):
        """
            parameters
        """
        if not(isinstance(img_npy, np.array)):
            raise TypeError("img_npy should be of type np.array")
        
        print(img_npy.dtype)



    
