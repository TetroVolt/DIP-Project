
import tkinter as TK
import tkinter.filedialog
from tkinter import ttk
from . import ImageViewer

# import the Transformations library
from .Library import lib

class App:
    def __init__(self, master, img=None):
        self.master = master
        self.tool_box = ttk.Frame(master, padding=20)
        self.tool_box.grid()

        self.__initButtons()
        self.viewer = None

    def __initButtons(self):
        self.openFileButton = ttk.Button(
            self.tool_box, 
            text="open file",
            command=self.openFileButtonPressed)
        self.openFileButton.grid(column=0)

        self.rotateButton = ttk.Button(
            self.tool_box,
            text="Rotate",
            command=self.rotateButtonPressed)
        self.rotateButton.grid(column=0)

    def openFileButtonPressed(self):
        filetypes = {
            "gif files":"*.gif",
            "jpeg files":"*.jpg",
            "png files":"*.png",
            "all files":"*.*"
        }
        filename = TK.filedialog.askopenfilename(
            initialdir = "./",
            title = "Select file",
            filetypes = list(filetypes.items()))

        if filename:
            self.viewer = ImageViewer.ImageViewer(self.master, filename)
            self.viewer.grid(row=0, column=1)
    
    def resizeButtonPressed(self):
        if self.viewer is None: return

    def rotateButtonPressed(self):
        if self.viewer is None: return
        self.viewer.rotate()

