#%%
import os
import numpy as np
from multiprocessing import cpu_count
from pathlib import Path, PurePath
import matplotlib.pyplot as plt
from itertools import product
from PIL import Image
import cv2

import torch
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from torchsummary import summary

from datasets.load_data import SudokuDataset, show_sudoku_batch
from models.unet import UNet
from sdks.classifier import SudokuClassifier
from train.train_callbacks import TensorboardVisualizerCallback, TensorboardLoggerCallback, ModelSaverCallback

# hyperparameters
# size = 512
# scale = 32
# batch_size = 2
# epochs = 120
# lr = 0.01
train_ratio = 0.8

parameters = dict(
    size = [512],
    scale = [8],
    batch_size = [1, 2],
    epochs = [100, 150, 200, 250, 300],
    lr = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
)

# system info
seg_data_dir = PurePath('data', 'segmentation_data')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
counter = 0

param_values = [v for v in parameters.values()]

for size, scale, batch_size, epochs, lr in product(*param_values):
    counter += 1
    info = f'[{counter}]size={size} scale={scale} bs={batch_size} e={epochs} lr={lr}'
    tb_viz_cb = TensorboardVisualizerCallback(Path('runs/images/' + info))
    tb_log_cb = TensorboardLoggerCallback(Path('runs/logs/' + info))
    model_saver_cb = ModelSaverCallback('output/models/model_' + str(counter), verbose=True)
    
    
    # get datasets
    transform = T.Compose([T.Resize((size, size)), T.CenterCrop(size), T.ToTensor()])
    sudoku_dataset = SudokuDataset(seg_data_dir, transform=transform)
    train_size =int(train_ratio*len(sudoku_dataset))
    val_size = len(sudoku_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(sudoku_dataset, [train_size, val_size])
    
    net = UNet((3, size, size), scale)
    classifier = SudokuClassifier(net, epochs)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    print('train size: {', len(train_loader.dataset), '}, val size: {', len(val_loader.dataset), '}')
    
    classifier.train(train_loader, val_loader, optimizer, epochs, callbacks=[tb_viz_cb, tb_log_cb, model_saver_cb])

# classifier.restore_model('data\seg_model')
# 
# image = Image.open(Path.joinpath(seg_data_dir, 'images', '0007_05.jpg').as_posix())
# img = transform(image).unsqueeze(0)
# mask = np.squeeze(classifier.predict_one(img))
# 
# image = plt.imread(Path.joinpath(seg_data_dir, 'images', '0007_05.jpg').as_posix())
# height, width, _ = image.shape
# mask = np.dstack((mask, mask, mask)) * np.array([255, 255, 255])
# mask = mask.astype(np.uint8)
# mask = cv2.resize(mask, (width, height))
# 
# result = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
# 
# mask = mask[:, :, 0]
# cv2.imwrite('mask_2.png', mask)
# 
# plt.imshow(result)
# plt.show()

# 5, 12, 18, 39, 46, 54, 56, 59

# %%
