#%%
from pathlib import Path
from time import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms, datasets

#%%

data_path = Path('data/Chars74K')

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
    ])

train_dataset = datasets.ImageFolder(
    root=data_path,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=10,
    shuffle=True
)

#%%

for batch_idx, (data, target) in enumerate(train_loader):
    print(data.shape)
    print(target.shape)
    break

# %%
print(target)


# %%
plt.imshow(data[4].view(128, 128), cmap='gray')
plt.show()

# %%
