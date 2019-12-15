#%%
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import cv2

from sdks.nn.vgg import VGG_like
import sdks.sudoku_engine as engine

#%%

def get_board_matrix(pads):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 10
    
    model = VGG_like(16).to(device)
    model.load_state_dict(torch.load('data/ocr_model'))
    model.eval()
    
    # show one pad
    # image = cv2.cvtColor(pads[3], cv2.COLOR_BGR2RGB)
    # print(image.sum())
    # # plt.imshow(image)
    
    board = []
    
    transform = T.Compose([
        T.ToTensor()
        ])
    
    for pad in pads:
        image = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB)
        image = transform(image).unsqueeze(dim=0).to(device)
        with torch.no_grad():
            pred = model(image)
        pred = np.argmax(pred.cpu().numpy())
        threshold = 9200
        # print(image.sum())
        if image.sum() > threshold:
            pred = 0
        board.append(pred)
    
    board = np.array(board).reshape(9, 9)
    
    return board

# %%
if __name__ == "__main__":
    with open('data/num_pads.txt', 'rb') as fp:
        pads = np.load(fp)
    
    board = get_board_matrix(pads)

    engine.solve(board)
    print(board)
