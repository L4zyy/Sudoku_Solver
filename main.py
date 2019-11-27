import numpy as np
from cv2 import cv2 as cv
import matplotlib.pyplot as plt
from sdks.solver import sdkSolver

# img = cv.imread('data/sudoku_dataset/images/image9.jpg')
# img = cv.imread('data/sudoku.jpg')

solver = sdkSolver()
solver.setup_new_img('data/test.png')
num_imgs = solver.get_num_imgs()

def show_num_pads():
    for i in range(9):
        for j in range(9):
            plt.subplot(9, 9, 9*i+j+1, frameon=False)
            plt.imshow(num_imgs[9*i+j], cmap='gray')
            plt.axis('off')
    plt.show()

board = solver.get_sudoku_board_matrix()
print(board)

answer = solver.solve(board)
if answer is not None:
    print(answer)
else:
    print('Board not valid.')