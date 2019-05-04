
import tkinter as tk
import numpy as np
from .Library import lib
from PIL import Image, ImageTk

class ImageViewer(tk.Label):
    """ 
    Responsible for rendering the transformed image
    """
    class State:
        def __init__(self, shape=None):
            self.zoom_mode = False
            self.x, self.y = 0, 0
            self.zoom_amt = 1
            self.prev_zoom_amt = 1
            self.last_clicked_x, self.last_clicked_y = 0,0
            self.set_default_homology(shape)
            self.affinepad = False
            self.interpolation = lib.INTER_NEAREST
            
        def set_default_homology(self, shape=None):
            if shape:
                h, w = shape
                self.original_homology = np.array([
                    [0, w, w, 0],
                    [0, 0, h, h],
                    [1, 1, 1, 1]
                ], dtype=np.float)
            else:
                self.original_homology = None

    def __init__(self, master, filenm: str, *args, **kwargs):
        self.setImage(filenm)
        super().__init__(master, image=self.photoTK, *args, **kwargs)

        self.bind('<Enter>', self.mouseEnter)
        self.bind('<Leave>', self.mouseLeave)
        self.bind('<Motion>', self.mouseMotion)
        self.bind('<Button-1>', self.mouseBtn1)
        self.bind('<Button-3>', self.mouseBtn3)

    def setImage(self, filenm: str):
        self.photo = Image.open(filenm).convert('L')
        self.np_photo = np.array(self.photo)
        self.photoTK = ImageTk.PhotoImage(self.photo)
        self.state = ImageViewer.State(self.np_photo.shape)
        self.zoomed_img = self.np_photo
        assert(len(self.np_photo.shape)==2)

    def setImageFromNPArray(self, nparray: np.ndarray):
        assert(isinstance(nparray, np.ndarray))
        self.np_photo = nparray
        self.state.bound_top_left = 0,0
        self.state.bound_bottom_right = self.np_photo.shape[::-1]
        self.photo = Image.fromarray(self.np_photo)
        self.photoTK = ImageTk.PhotoImage(self.photo)
        #self.state = ImageViewer.State(self.np_photo.shape)

    def setInterpolation(self, inter):
        choices = {
            'NEAREST':lib.INTER_NEAREST, 
            'LINEAR': lib.INTER_LINEAR,
            'CUBIC': lib.INTER_CUBIC,
            'LANCZOS4': lib.INTER_LANCZOS4 }
        
        self.state.interpolation = choices[inter]

    def recalculateImageBounds(self):
        """
        MUST TRY TO DO THIS ASYNCHRONOUSLY
        """
        zoom_amt = self.state.zoom_amt
        if zoom_amt == 1: 
            self.zoomed_img = self.np_photo
        else:
            label_h, label_w = self.np_photo.shape

            zoom_w, zoom_h = label_w // zoom_amt, label_h // zoom_amt
            if min(zoom_w, zoom_h) < 8: # ensure that lowest dimension is not less than 8 pixels
                zoom_amt = self.state.zoom_amt = self.state.prev_zoom_amt
                zoom_w, zoom_h = label_w // zoom_amt, label_h // zoom_amt

            self.zoomed_img = lib.resize(
                        src=self.np_photo,
                        dsize=(label_w * self.state.zoom_amt, label_h * self.state.zoom_amt),
                        interpolation=self.state.interpolation)

    def display_zoom_picture(self):
        X,Y = self.state.x, self.state.y
        vp_rows, vp_cols = self.np_photo.shape # view port rows and cols
        margin_vert, margin_horz = vp_rows // 2, vp_cols // 2

        zoom_amt = self.state.zoom_amt
        X *= zoom_amt
        Y *= zoom_amt

        X = max(margin_horz, min(X, zoom_amt*vp_cols - margin_horz))
        Y = max(margin_vert, min(Y, zoom_amt*vp_rows - margin_vert))

        temp = self.zoomed_img[Y-margin_vert:Y+margin_vert, X-margin_horz:X+margin_horz]

        self.photo = Image.fromarray(temp)
        self.photoTK = ImageTk.PhotoImage(self.photo)
        self.configure(image=self.photoTK)

    def mouseMotion(self, event):
        self.state.x, self.state.y = event.x, event.y
        if self.state.zoom_mode:
            self.display_zoom_picture()
        
    def mouseEnter(self, event):
        pass
    
    def mouseLeave(self, event):
        self.photo = Image.fromarray(self.np_photo)
        self.photoTK = ImageTk.PhotoImage(self.photo)
        self.configure(image=self.photoTK)

    def mouseBtn1(self, event):
        self.state.last_clicked_x, self.state.last_clicked_y = event.x, event.y
        print('last clicked:', event.x, event.y)
        if self.state.zoom_mode:
            self.state.prev_zoom_amt = self.state.zoom_amt
            self.state.zoom_amt *= 2
        self.recalculateImageBounds()
        self.display_zoom_picture()
    
    def mouseBtn3(self, event):
        if self.state.zoom_mode:
            self.state.prev_zoom_amt = self.state.zoom_amt
            self.state.zoom_amt = max(1,self.state.zoom_amt // 2)
        self.recalculateImageBounds()
        self.display_zoom_picture()

    def saveImage(self, filename:str):
        import scipy.misc
        scipy.misc.imsave(filename, self.np_photo)

    def fisheye(self):
        temp = lib.fisheye(self.np_photo)
        self.setImageFromNPArray(temp)
        self.configure(image=self.photoTK)

    def warpPerspective(self, mat: np.array):
        rows, cols = self.np_photo.shape
        temp = lib.warpPerspective(self.np_photo, mat, (cols,rows), self.state.interpolation)
        self.state.set_default_homology(self.np_photo.shape)
        self.setImageFromNPArray(temp)
        self.configure(image=self.photoTK)

    def affineTransform(self, mat: np.array):
        rows, cols = self.np_photo.shape
        temp = lib.warpAffine(self.np_photo, mat, (cols,rows), self.state.interpolation)
        self.state.set_default_homology(self.np_photo.shape)
        self.setImageFromNPArray(temp)
        self.configure(image=self.photoTK)

    def paddedAffineTransform(self, mat: np.array):
        rows, cols = self.np_photo.shape

        mapped_homologies = mat.dot(self.state.original_homology)
        x_coords = mapped_homologies[0,:]
        y_coords = mapped_homologies[1,:]
        min_x, max_x = x_coords.min(), x_coords.max()
        min_y, max_y = y_coords.min(), y_coords.max()

        dst_shape = (int(np.round(max_x - min_x)), int(np.round(max_y - min_y)))
            
        # add translation to the transformation matrix to shift to positive values
        mat[0,2] -= min_x
        mat[1,2] -= min_y

        temp = lib.warpAffine(self.np_photo, mat, dst_shape, self.state.interpolation)
        self.setImageFromNPArray(temp)
        self.configure(image=self.photoTK)

        mapped_homologies[0,:] -= min_x
        mapped_homologies[1,:] -= min_y
        self.state.original_homology = np.vstack((mapped_homologies,[1,1,1,1]))

