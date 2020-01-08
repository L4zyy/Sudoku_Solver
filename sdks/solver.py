import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torchvision
from torch import nn as nn
from torchvision import transforms as T

import cv2
from PIL import Image

from models.unet import UNet
from models.vgg import VGG_like
from sdks.classifier import SudokuClassifier
from sdks.get_board_image import get_board_image
from sdks.get_board_matrix import get_board_matrix
import sdks.sudoku_engine as engine


class sdkSolver():
    def __init__(self, seg_model_path, ocr_model_path):
        self.seg_model = UNet((3, 512, 512), 32)
        self.ocr_model = VGG_like(16)
        
        self.seg_model.load_state_dict(torch.load(seg_model_path))
        self.ocr_model.load_state_dict(torch.load(ocr_model_path))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seg_model.to(self.device)
        self.ocr_model.to(self.device)

        self.current_image = None
        self.current_image = None
    
    def solve(self, image):
        classifier = SudokuClassifier(self.seg_model, 0)
        classifier.restore_model('data\seg_model')

        size = 512
        input_img_resize = (size, size)
        transform = T.Compose([T.Resize(input_img_resize), T.CenterCrop(size), T.ToTensor()])
        img = transform(image).unsqueeze(0)
        mask = np.squeeze(classifier.predict_one(img))

        width, height = image.size
        mask_mix = np.dstack((mask, mask, mask)) * np.array([255, 255, 255])
        mask_mix = mask_mix.astype(np.uint8)
        mask_mix = cv2.resize(mask_mix, (width, height))

        # comvert image from PIL to Array
        image = np.array(image)
        result = cv2.addWeighted(mask_mix, 0.5, image, 0.5, 0.)
        # convert mask to array
        mask = np.array(mask).astype(np.uint8)
        mask = cv2.resize(np.array(mask), (width, height))

        # mask = mask[:, :, 0]
        # cv2.imwrite('mask_2.png', mask)

        # plt.imshow(result)
        # plt.show()

        # image = cv2.imread('data/image_2.jpg')
        # mask = cv2.imread('data/mask_2.png', cv2.IMREAD_GRAYSCALE)
        print(mask.shape)

        pads, warp, retransform, contour = get_board_image(image, mask, True)
    
        # with open('data/num_pads.txt', 'wb') as fp:
        #     np.save(fp, pads)

        puzzle = get_board_matrix(pads)
        board = np.copy(puzzle)

        engine.solve(board)
        print(board)

        pad_size = 64

        for i in range(9):
            for j in range(9):
                if puzzle[i][j] == 0:
                    y = (i+1)*pad_size - 16
                    x = j*pad_size + 16
                    warp = cv2.putText(warp, str(board[i][j]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        re_img = cv2.warpPerspective(warp, retransform, (width, height))
        image = image[:, :, ::-1]
        print(re_img.shape)
        bg_mask = np.ones_like(image)
        bg_mask = cv2.drawContours(bg_mask, [contour], 0, (255, 255, 255), -1)
        bg_mask = cv2.bitwise_not(bg_mask)
        background = cv2.bitwise_and(image, bg_mask)
        re_img = cv2.add(re_img, background)
        
        cv2.imshow('answer', warp)
        cv2.imshow('background', background)
        cv2.imshow('retransform', re_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def print(self):
        pass

if __name__ == "__main__":
    image = Image.open('data/image_2.jpg')
    solver = sdkSolver('data/seg_model', 'data/ocr_model')

    solver.solve(image)