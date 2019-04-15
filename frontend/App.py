
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

        self.rotateButton = ttk.Button(
            self.frame,
            text="Resize",
            command=self.resizeButtonPressed)
        self.rotateButton.grid(column=0)

        self.rotateButton = ttk.Button(
            self.frame,
            text="Rotate",
            command=self.rotateButtonPressed)
        self.rotateButton.grid(column=0)

        self.rotateButton = ttk.Button(
            self.frame,
            text="Affine",
            command=self.affineButtonPressed)
        self.rotateButton.grid(column=0)

    def openFileButtonPressed(self):
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
    
    def saveFileButtonPressed(self):
        filename = tk.filedialog.asksaveasfilename(
                        initialdir = "./",
                        title = "Select file",
                        filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

        if filename and self.viewer:
            self.viewer.saveImage(filename)

    def resizeButtonPressed(self):
        if self.viewer is None: return
        # TODO open dialog to ask how much in x and y to resize by

    def rotateButtonPressed(self):
        if self.viewer is None: return
        # TODO open dialog box to ask for theta and center of rotation

    def affineButtonPressed(self):
        if self.viewer is None: return
        # TODO open dialog box for user to enter matrix values manually
        # for i hat, j hat, and offsets
        self.viewer.affineTransform(np.zeros(1))


