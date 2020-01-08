#%%
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import cv2

from models.vgg import VGG_like
import sdks.sudoku_engine as engine

#%%

def get_board_matrix(pads):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 10
    
    model = VGG_like(16).to(device)
    model.load_state_dict(torch.load('data/ocr_model'))
    model.eval()
    
    # show some pads
    fig = plt.figure()
    image = cv2.cvtColor(pads[0], cv2.COLOR_BGR2RGB)
    plt.subplot(2, 4, 1)
    plt.axis('off')
    plt.imshow(image)
    image = cv2.cvtColor(pads[1], cv2.COLOR_BGR2RGB)
    plt.subplot(2, 4, 2)
    plt.axis('off')
    plt.imshow(image)
    image = cv2.cvtColor(pads[2], cv2.COLOR_BGR2RGB)
    plt.subplot(2, 4, 3)
    plt.axis('off')
    plt.imshow(image)
    image = cv2.cvtColor(pads[3], cv2.COLOR_BGR2RGB)
    plt.subplot(2, 4, 4)
    plt.axis('off')
    plt.imshow(image)
    image = cv2.cvtColor(pads[4], cv2.COLOR_BGR2RGB)
    plt.subplot(2, 4, 5)
    plt.axis('off')
    plt.imshow(image)
    image = cv2.cvtColor(pads[5], cv2.COLOR_BGR2RGB)
    plt.subplot(2, 4, 6)
    plt.axis('off')
    plt.imshow(image)
    image = cv2.cvtColor(pads[6], cv2.COLOR_BGR2RGB)
    plt.subplot(2, 4, 7)
    plt.axis('off')
    plt.imshow(image)
    image = cv2.cvtColor(pads[10], cv2.COLOR_BGR2RGB)
    plt.subplot(2, 4, 8)
    plt.axis('off')
    plt.imshow(image)

    plt.show()
    
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


# %%
