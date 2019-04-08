
import tkinter as TK
from tkinter import ttk

# import the Transformations library
from .Library import TransformFunctions as lib

class App:
    def __init__(self, master):
        self.tool_box = ttk.Frame(master, padding=20)
        self.tool_box.grid()

        self.zoomInButton = ttk.Button(
            self.tool_box, 
            text="Zoom+",
            command=self.zoomInButtonPressed)
        self.zoomInButton.grid(column=0)

        self.zoomOutButton = ttk.Button(
            self.tool_box,
            text="Zoom-",
            command=self.zoomOutButtonPressed)
        self.zoomOutButton.grid(column=0)

        self.resizeButton = ttk.Button(
            self.tool_box,
            text="Resize",
            command=self.resizeButtonPressed)
        self.resizeButton.grid(column=0)

        self.rotateButton = ttk.Button(
            self.tool_box,
            text="Rotate",
            command=self.rotateButtonPressed)
        self.rotateButton.grid(column=0)

        self.photo = TK.PhotoImage(file='./kennysmall.gif')
        self.photoLabel = TK.Label(master, image=self.photo, padx=80, pady=80)
        self.photoLabel.grid(row=0,column=1)

    def zoomInButtonPressed(self):
        print("Zoom In Button pressed")

    def zoomOutButtonPressed(self):
        print("Zoom Out Button pressed")

    def resizeButtonPressed(self):
        print("resize Button pressed")

    def rotateButtonPressed(self):
        print("rotate Button pressed")

