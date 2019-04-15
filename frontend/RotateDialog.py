import tkinter as tk
from tkinter import ttk
import numpy as np

class RotateDialog(tk.Tk):
    def __init__(self, img=None):
        super().__init__()
        self.__initButtons()

    def __initButtons(self):
        self.rotateButton = ttk.Button(
            self.frame,
            text="Affine",
            command=self.affineButtonPressed)
        self.rotateButton.grid(column=0)

    def rotateButtonPressed(self):
        if self.viewer is None: return
        # TODO open dialog box to ask for theta and center of rotation

