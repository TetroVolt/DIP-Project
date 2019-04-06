
import tkinter as TK
from ImageObj import ImageObj

class ImageViewer:
    """ 
    Responsible for rendering the transformed image
    """
    def __init__(self, img: np.array):
        self.imgObj = ImageObj(img)
