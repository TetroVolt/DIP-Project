
import tkinter as TK

# import the Transformations library
from .Library import Library
Library = Library(source="opencv")

class App:
    def __init__(self, master):
        frame = TK.Frame(master)
        frame.grid()

        self.button = TK.Button(
            frame, text="QUIT", fg="red", 
            command=frame.quit)
        self.button.pack()

        self.hi_there = TK.Button(
            frame, text="hello", 
            command=self.say_hi)

        self.hi_there.pack(side=TK.LEFT)

    def say_hi(self):
        print('Hello')

