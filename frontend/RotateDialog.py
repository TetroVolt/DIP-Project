import tkinter as tk
from tkinter import ttk
import numpy as np

class RotateDialog(tk.Tk):
    def __init__(self, img=None):
        super().__init__()
        self.attributes('-type', 'dialog')
        self.frame = ttk.Frame(self, padding=20)
        self.frame.grid()
        self.__initButtons()

    def __initButtons(self):
        self.rotateButton = ttk.Button(
            self.frame,
            text="Rotate",
            command=self.rotateButtonPressed)
        self.rotateButton.grid(column=0)

    def rotateButtonPressed(self):
        self.destroy()
        pass

