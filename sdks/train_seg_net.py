import os
from multiprocessing import cpu_count
from pathlib import Path, PurePath
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from torchsummary import summary

from load_data import SudokuDataset, show_sudoku_batch
from nn.unet import UNet
import classifier
from train_callbacks import TensorboardVisualizerCallback, TensorboardLoggerCallback

# hyperparameters
size = 512
input_img_resize = (size, size)
batch_size = 2
epochs = 100
lr = 0.01
train_ratio = 0.8
sample_size = None

seg_data_dir = PurePath('data', 'segmentation_data')

# optional parameters
threads = cpu_count()
use_cuda = torch.cuda.is_available()

tb_viz_cb = TensorboardVisualizerCallback(Path('logs/images'))
tb_log_cb = TensorboardLoggerCallback(Path('logs'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0

# summary(model, (3, 512, 512))

transform = T.Compose([T.Resize(input_img_resize), T.CenterCrop(size), T.ToTensor()])

sudoku_dataset = SudokuDataset(seg_data_dir, transform=transform)

train_size =int(train_ratio*len(sudoku_dataset))
val_size = len(sudoku_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(sudoku_dataset, [train_size, val_size])

net = UNet((3, 512, 512))
classifier = classifier.SudokuClassifier(net, epochs)
optimizer = optim.Adam(net.parameters(), lr=lr)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

print('train size: {', len(train_loader.dataset), '}, val size: {', len(val_loader.dataset), '}')

classifier.train(train_loader, val_loader, optimizer, epochs, callbacks=[tb_viz_cb, tb_log_cb])