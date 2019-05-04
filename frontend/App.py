
import tkinter as tk
import tkinter.filedialog
from tkinter import ttk
from . import ImageViewer
import numpy as np

# import the Transformations library
from .Library import lib

class App(tk.Tk):
    def __init__(self, img=None):
        super().__init__()
        self.frame = ttk.Frame(self, padding=20)
        self.frame.grid()
        self.__initButtons()
        self.viewer = None

    def __initButtons(self):
        self.openFileButton = ttk.Button(
            self.frame, 
            text="open file",
            command=self.openFileButtonPressed)
        self.openFileButton.grid(column=0)

        self.openFileButton = ttk.Button(
            self.frame, 
            text="save file",
            command=self.saveFileButtonPressed)
        self.openFileButton.grid(column=0)

        self.resizeButton = ttk.Button(
            self.frame,
            text="Resize",
            command=self.resizeButtonPressed)
        self.resizeButton.grid(column=0)

        self.rotateButton = ttk.Button(
            self.frame,
            text="Rotate",
            command=self.rotateButtonPressed)
        self.rotateButton.grid(column=0)

        self.affineButton = ttk.Button(
            self.frame,
            text="Affine",
            command=self.affineButtonPressed)
        self.affineButton.grid(column=0)

        self.paddingButton = ttk.Button(
            self.frame,
            text="padding",
            command=self.paddingButtonPressed)
        self.paddingButton.grid(column=0)

        self.perspectiveButton = ttk.Button(
            self.frame,
            text="perspective",
            command=self.perspectiveButtonPressed)
        self.perspectiveButton.grid(column=0)

        self.fisheyeButton = ttk.Button(
            self.frame,
            text="fisheye",
            command=self.fisheyeButtonPressed)
        self.fisheyeButton.grid(column=0)

        self.zoomButton = ttk.Button(
            self.frame,
            text="Zoom",
            command=self.zoomButtonPressed)
        self.zoomButton.grid(column=0)

        self.interpolationMenu = tk.StringVar(self.frame)
        choices = ['NEAREST', 'LINEAR' , 'CUBIC' , 'LANCZOS4']
        self.interOptionMenu = ttk.OptionMenu(self.frame, self.interpolationMenu, "Choose Interpolation", *choices)
        self.interOptionMenu.grid(column=0)
        self.interOptionMenu.configure(state='disabled')
        self.interpolationMenu.trace('w',self.interpolationMenuChoice)

    def openFileButtonPressed(self):
        """
        opens open file dialog to choose an image
        """
        filetypes = {
            "all files":"*.*",
            "gif files":"*.gif",
            "jpeg files":"*.jpg",
            "png files":"*.png",
        }
        filename = tk.filedialog.askopenfilename(
            initialdir = "./",
            title = "Select file",
            filetypes = list(filetypes.items()))

        if filename:
            if self.viewer is None:
                self.viewer = ImageViewer.ImageViewer(self, filename)
            else:
                self.viewer.setImage(filename)
            self.viewer.grid(row=0, column=1)

            self.interOptionMenu.configure(state='enabled')
    
    def saveFileButtonPressed(self):
        """ 
        opens save file dialog to save image
        """
        if self.viewer is None: return

        filename = tk.filedialog.asksaveasfilename(
                        initialdir = "./",
                        title = "Select file",
                        filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

        if filename and self.viewer:
            self.viewer.saveImage(filename)

    def perspectiveButtonPressed(self):
        """
        creates configuration dialog to ask for 3 points before and after. 
        Uses the points to create an affine matrix to pass onto the viewer
        to transform the image
        """
        if self.viewer is None: return
        # TODO open dialog box for user to enter matrix values manually
        # for i hat, j hat, and offsets

        self.viewer.state.zoom_mode = False
        center = self.viewer.state.last_clicked_x, self.viewer.state.last_clicked_y

        root = tk.Tk()
        root.attributes('-type', 'dialog')
        frame = ttk.Frame(root, padding=20)
        frame.grid()

        pt1_start = tk.Entry(frame)
        pt1_final = tk.Entry(frame)
        pt2_start = tk.Entry(frame)
        pt2_final = tk.Entry(frame)
        pt3_start = tk.Entry(frame)
        pt3_final = tk.Entry(frame)
        pt4_start = tk.Entry(frame)
        pt4_final = tk.Entry(frame)

        pt1_start.grid(row=1, column=0)
        pt1_final.grid(row=1, column=2)
        pt2_start.grid(row=2, column=0)
        pt2_final.grid(row=2, column=2)
        pt3_start.grid(row=3, column=0)
        pt3_final.grid(row=3, column=2)
        pt4_start.grid(row=4, column=0)
        pt4_final.grid(row=4, column=2)
        tk.Label(frame, text="enter 4 points before and after transform\npoints given in form: num,num").grid(row=0, column=0)
        tk.Label(frame, text=">").grid(row=2, column=1)

        def doneAction():
            pt1_s, pt1_f = pt1_start.get(), pt1_final.get()
            pt2_s, pt2_f = pt2_start.get(), pt2_final.get()
            pt3_s, pt3_f = pt3_start.get(), pt3_final.get()
            pt4_s, pt4_f = pt4_start.get(), pt4_final.get()

            pts = [pt1_s, pt2_s, pt3_s, pt4_s, pt1_f, pt2_f, pt3_f, pt4_f]

            try:
                pts = [[float(c) for c in pt.split(',')] for pt in pts]
                if (any(len(pt)!=2 for pt in pts)): return
            except:
                print("invalid pts")
                return

            pt1_s, pt2_s, pt3_s, pt4_s, pt1_f, pt2_f, pt3_f, pt4_f = pts
            start = np.float32([pt1_s, pt2_s, pt3_s, pt4_s])
            end = np.float32([pt1_f, pt2_f, pt3_f, pt4_f])
            mat = lib.getPerspectiveTransform(start, end)

            self.viewer.warpPerspective(mat)

            root.destroy()

        done = tk.Button(frame, text="done", command=doneAction)
        done.grid()

    def resizeButtonPressed(self):
        """
        creates dialog box to ask for fx and fy to resize image by
        """
        if self.viewer is None: return
        # TODO open dialog to ask how much in x and y to resize by
        self.viewer.state.zoom_mode = False
        center = self.viewer.state.last_clicked_x, self.viewer.state.last_clicked_y

        root = tk.Tk()
        root.attributes('-type', 'dialog')

        frame = ttk.Frame(root, padding=20)
        frame.grid()

        fx_entry = tk.Entry(frame)
        fx_entry.grid(row=0, column=1)
        fy_entry = tk.Entry(frame)
        fy_entry.grid(row=1, column=1)

        def doneAction():
            fx, fy = fx_entry.get(), fy_entry.get()
            try:
                fx, fy = float(fx), float(fy)
            except:
                return
            self.viewer.resizeImage(fx, fy)
            root.destroy()

        tk.Button(frame, text="done", command=doneAction).grid(row=2) # done button
        tk.Label(frame, text="scale_x: ").grid(row=0)
        tk.Label(frame, text="scale_y: ").grid(row=1)

    def rotateButtonPressed(self):
        """
        creates dialog box to ask for theta to rotate image by, 
        center of rotation used is the last clicked coordinate 
        on the image.
        """
        if self.viewer is None: return
        self.viewer.state.zoom_mode = False
        center = self.viewer.state.last_clicked_x, self.viewer.state.last_clicked_y

        root = tk.Tk()
        root.attributes('-type', 'dialog')
        frame = ttk.Frame(root, padding=20)
        frame.grid()
        tk.Label(frame, text="Rotation will be done about the last clicked pixel on the image\nEnter the amount in Degrees:").grid()
        deg_entry = tk.Entry(frame)
        deg_entry.grid()

        def doneAction():
            degrees = deg_entry.get()
            try:
                degrees = int(degrees)
            except:
                return
            mat = lib.getRotationMatrix2D(center, degrees, 1)
            if self.viewer.state.affinepad:
                self.viewer.paddedAffineTransform(mat)
            else:
                self.viewer.affineTransform(mat)

            root.destroy()
        done = tk.Button(frame, text="done", command=doneAction)
        done.grid()


    def affineButtonPressed(self):
        """
        creates configuration dialog to ask for 3 points before and after. 
        Uses the points to create an affine matrix to pass onto the viewer
        to transform the image
        """
        if self.viewer is None: return
        # TODO open dialog box for user to enter matrix values manually
        # for i hat, j hat, and offsets

        self.viewer.state.zoom_mode = False
        center = self.viewer.state.last_clicked_x, self.viewer.state.last_clicked_y

        root = tk.Tk()
        root.attributes('-type', 'dialog')
        frame = ttk.Frame(root, padding=20)
        frame.grid()

        pt1_start = tk.Entry(frame)
        pt1_final = tk.Entry(frame)
        pt2_start = tk.Entry(frame)
        pt2_final = tk.Entry(frame)
        pt3_start = tk.Entry(frame)
        pt3_final = tk.Entry(frame)

        pt1_start.grid(row=1, column=0)
        pt1_final.grid(row=1, column=2)
        pt2_start.grid(row=2, column=0)
        pt2_final.grid(row=2, column=2)
        pt3_start.grid(row=3, column=0)
        pt3_final.grid(row=3, column=2)
        tk.Label(frame, text="enter 3 points before and after transform\npoints given in form: num,num").grid(row=0, column=0)
        tk.Label(frame, text=">").grid(row=2, column=1)

        def doneAction():
            pt1_s, pt1_f = pt1_start.get(), pt1_final.get()
            pt2_s, pt2_f = pt2_start.get(), pt2_final.get()
            pt3_s, pt3_f = pt3_start.get(), pt3_final.get()

            pts = [pt1_s, pt2_s, pt3_s, pt1_f, pt2_f, pt3_f]

            try:
                pts = [[float(c) for c in pt.split(',')] for pt in pts]
                if (any(len(pt)!=2 for pt in pts)): return
            except:
                return

            pt1_s, pt2_s, pt3_s, pt1_f, pt2_f, pt3_f = pts
            start = np.float32([pt1_s, pt2_s, pt3_s])
            end = np.float32([pt1_f, pt2_f, pt3_f])
            mat = lib.getAffineTransform(start, end)
            if self.viewer.state.affinepad:
                self.viewer.paddedAffineTransform(mat)
            else:
                self.viewer.affineTransform(mat)

            root.destroy()

        done = tk.Button(frame, text="done", command=doneAction)
        done.grid()

    def zoomButtonPressed(self):
        """
        toggles zoom mode for hover zoom on the image
        """
        if self.viewer is None: return
        self.viewer.state.zoom_mode = not self.viewer.state.zoom_mode
        relief = ['pressed'] if self.viewer.state.zoom_mode else ['!pressed']
        self.zoomButton.state(relief)
    
    def paddingButtonPressed(self):
        """
        toggles that affinepad variable in the viewer state object
        """
        if self.viewer is None: return
        self.viewer.state.affinepad = not self.viewer.state.affinepad
        relief = ['pressed'] if self.viewer.state.affinepad else ['!pressed']
        self.paddingButton.state(relief)

    def fisheyeButtonPressed(self):
        """
        opens dialog for fisheye configuration and applies fish eye transform
        """
        #TODO
        if self.viewer is None: return
        self.viewer.fisheye()

    def interpolationMenuChoice(self, *args):
        #print(self.interpolationMenu.get())
        if self.viewer is None: return
        interpolation = self.interpolationMenu.get()
        self.viewer.setInterpolation(interpolation)