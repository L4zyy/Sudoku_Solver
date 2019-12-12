#%%
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import cv2

from nn.vgg import VGG_like

#%%


with open('data/num_pads.txt', 'rb') as fp:
    pads = np.load(fp)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_classes = 10

model = VGG_like(16).to(device)
model.load_state_dict(torch.load('data/num_model'))
model.eval()

# %%

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
image = cv2.filter2D(pads[0], -1, kernel)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)

# %%

board = []

transform = T.Compose([
    T.ToTensor()
    ])

for pad in pads:
    image = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(dim=0).to(device)
    with torch.no_grad():
        pred = model(image)
    # threshold = 1000
    pred = np.argmax(pred.cpu().numpy())
    board.append(pred)

board = np.array(board).reshape(9, 9)
print(board)


# %%
image = cv2.cvtColor(pads[0], cv2.COLOR_BGR2RGB)
image = transform(image).unsqueeze(dim=0).to(device)
with torch.no_grad():
    pred = model(image)
pred = np.argmax(pred.cpu().numpy())
print(pred)

# %%
