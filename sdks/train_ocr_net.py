import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import helpers

# Hyperparameters
train_ratio = 0.8
batch_size = 64
lr = 1e-4
epochs = 25
n_inputs = 4096
n_classes = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_data_path = Path('data/number_data')
        
transform = T.Compose([
    T.ToTensor()
    ])
        
num_dataset = datasets.ImageFolder(
    root=num_data_path,
    transform=transform
)

train_size =int(train_ratio*len(num_dataset))
val_size = len(num_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(num_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

print('train size: {', len(train_loader.dataset), '}, val size: {', len(val_loader.dataset), '}')

model = torchvision.models.vgg16_bn(pretrained=False)
model.classifier = nn.Sequential(
    nn.Linear(25088, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(64, n_classes),
    nn.LogSoftmax(dim=1)
)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


@helpers.timer
def train():
    tb = SummaryWriter()

    for epoch in range(epochs):
        # train
        total_loss = 0
        total_correct = 0

        it_num = len(train_loader)

        with tqdm(
            total=it_num,
            desc="Epochs {}/{}".format(epoch + 1, epochs),
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]'
        ) as pbar:
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                preds = model(images)
                loss = F.cross_entropy(preds, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_correct += get_num_correct(preds, labels)

                # update pbar
                pbar.set_postfix(OrderedDict(total_loss='{0:1.5f}'.format(total_loss),
                                             total_correct='{0:d}'.format(total_correct)))
                pbar.update(1)
        
        tb.add_scalar('(train) Total Loss', total_loss, epoch)
        tb.add_scalar('(train) Number of Correct', total_correct, epoch)
        tb.add_scalar('(train) Accuracy', total_correct / len(train_dataset), epoch)
        
        # validation
        total_loss = 0
        total_correct = 0

        it_num = len(val_loader)

        with tqdm(total=it_num, desc="Validating", leave=False) as pbar:
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    preds = model(images)
                loss = F.cross_entropy(preds, labels)

                total_loss += loss.item()
                total_correct += get_num_correct(preds, labels)

                pbar.update(1)
    
        print("Validation [total loss: {}, total correct: {}]".format(total_loss, total_correct))

        tb.add_scalar('(validation) Total Loss', total_loss, epoch)
        tb.add_scalar('(validation) Number of Correct', total_correct, epoch)
        tb.add_scalar('(validation) Accuracy', total_correct / len(val_dataset), epoch)
    tb.close()
        
train()
torch.save(model.state_dict(), 'data/num_model')