
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
            command=self.zoomInPressed) # zoom in add zoom out
        self.zoomInButton.grid(column=0)

        self.zoomOutButton = ttk.Button(
            self.tool_box,
            text="Zoom-",
            command=self.zoomOutPressed) # zoom in add zoom out
        self.zoomOutButton.grid(column=0)

        self.photo = TK.PhotoImage(file='./kennysmall.gif')
        self.photoLabel = TK.Label(master, image=self.photo, padx=80, pady=80)
        self.photoLabel.grid(row=0,column=1)

    def zoomInPressed(self):
        """
            gets called when zoom_button is pressed
        """
        print("Zoom In Button pressed")


    def zoomOutPressed(self):
        """
            gets called when zoom_button is pressed
        """
        print("Zoom Out Button pressed")

    
