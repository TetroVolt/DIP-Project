
import tkinter as TK
import numpy as np
from PIL import Image, ImageTk
from .Library import lib

class ImageViewer(TK.Label):
    """ 
    Responsible for rendering the transformed image
    """
    class ZoomState:
        def __init__(self):
            self.hovered = False
            self.x, self.y = 0, 0
            self.zoom_amt = 2
            self.prev_zoom_amt = 1
            pass

    def __init__(self, master, filenm: str, *args, **kwargs):
        self.setImage(filenm)
        super().__init__(master, image=self.photoTK, *args, **kwargs)

        self.bind('<Enter>', self.mouseEnter)
        self.bind('<Leave>', self.mouseLeave)
        self.bind('<Motion>', self.mouseMotion)
        self.bind('<Button-1>', self.mouseBtn1)
        self.bind('<Button-3>', self.mouseBtn3)

    def setImage(self, filenm):
        self.photo = Image.open(filenm).convert('L')
        self.np_photo = np.array(self.photo)
        self.photoTK = ImageTk.PhotoImage(self.photo)
        self.zoom_state = ImageViewer.ZoomState()

    def recalculateImageBounds(self):
        """
        MUST TRY TO DO THIS ASYNCHRONOUSLY
        """
        label_h, label_w = self.np_photo.shape
        zoom_amt = self.zoom_state.zoom_amt
        if zoom_amt == 1: return

        zoom_w, zoom_h = label_w // zoom_amt, label_h // zoom_amt
        if min(zoom_w, zoom_h) < 8: # ensure that lowest dimension is not less than 8 pixels
            zoom_amt = self.zoom_state.zoom_amt = self.zoom_state.prev_zoom_amt
            zoom_w, zoom_h = label_w // zoom_amt, label_h // zoom_amt

        # check if x, y of mouse is within safe bounds
        safe_min_x, safe_min_y = zoom_w // 2, zoom_h // 2
        safe_max_x, safe_max_y = label_w - safe_min_x, label_h - safe_min_y

        X,Y = self.zoom_state.x, self.zoom_state.y
        X = min(max(safe_min_x, X), safe_max_x)
        Y = min(max(safe_min_y, Y), safe_max_y)
        temp =  self.np_photo[Y-safe_min_y:Y+safe_min_y,X-safe_min_x:X+safe_min_x]
        temp = lib.resize(
                temp,
                dsize=(label_w, label_h),
                interpolation=lib.INTER_NEAREST)

        self.photo = Image.fromarray(temp)
        self.photoTK = ImageTk.PhotoImage(self.photo)
        self.configure(image=self.photoTK)

    def mouseMotion(self, event):
        self.zoom_state.x = event.x
        self.zoom_state.y = event.y
        self.recalculateImageBounds()
        #print(self.zoom_state, event)

    def mouseEnter(self, event):
        self.zoom_state.hover = True
    
    def mouseLeave(self, event):
        self.zoom_state.hover = False
        self.photo = Image.fromarray(self.np_photo)
        self.photoTK = ImageTk.PhotoImage(self.photo)
        self.configure(image=self.photoTK)

    def mouseBtn1(self, event):
        self.zoom_state.prev_zoom_amt = self.zoom_state.zoom_amt
        self.zoom_state.zoom_amt *= 2
        self.recalculateImageBounds()
    
    def mouseBtn3(self, event):
        self.zoom_state.prev_zoom_amt = self.zoom_state.zoom_amt
        self.zoom_state.zoom_amt = max(1,self.zoom_state.zoom_amt // 2)
        self.recalculateImageBounds()

    def saveImage(self, filename:str):
        import scipy.misc
        scipy.misc.imsave(filename, self.np_photo)

    def affineTransform(self, mat: np.array):
        pass



