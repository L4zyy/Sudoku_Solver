import numpy as np
from cv2 import cv2 as cv
import matplotlib.pyplot as plt
from sdks.solver import sdkSolver

from PIL import Image


image = Image.open('data/image_2.jpg')
solver = sdkSolver('data/seg_model', 'data/ocr_model')

solver.solve(image)