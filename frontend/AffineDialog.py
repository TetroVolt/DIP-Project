import tkinter as tk
from tkinter import ttk
import numpy as np

class AffineDialog(tk.Tk):
    def __init__(self, img=None):
        super().__init__()
        self.attributes('-type', 'dialog')
        self.frame = ttk.Frame(self, padding=20)
        self.frame.grid()
        self.__initButtons()

    def __initButtons(self):
        self.affineButton = ttk.Button(
            self.frame,
            text="Affine",
            command=self.affineButtonPressed)
        self.affineButton.grid(column=0)

    def affineButtonPressed(self):
        self.destroy()
        pass

