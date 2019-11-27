import numpy as np
from cv2 import cv2 as cv
import matplotlib.pyplot as plt
from sdks.solver import sdkSolver


solver = sdkSolver()
# solver.setup_new_img('data/sudoku_dataset/images/image1084.jpg')
# solver.setup_new_img('data/sudoku.jpg')
solver.setup_new_img('data/test.png')

solver.run()

if solver.valid:
    print(solver.solution)
    cv.imshow('output', solver.output_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print('Board not valid.')
    print(solver.board)