import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sdks.solver import sdkSolver

from PIL import Image


image = Image.open('assets/image_2.jpg')
solver = sdkSolver('data/seg_model', 'data/ocr_model')

solver.solve(image)