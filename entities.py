import numpy as np


class BoundingBox:
    """
    Boundary box of the word
    """
    x: int
    y: int
    w: int
    h: int

    def __init__(self,
                 x: int,
                 y: int,
                 w: int,
                 h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __str__(self):
        return f'(x, y, w, h) = ({self.x}, {self.y}, {self.w}, {self.h})'

    def area(self):
        return self.w * self.h


class Word:
    img: np.ndarray
    box: BoundingBox

    def __init__(self, img, box):
        self.img = img
        self.box = box
