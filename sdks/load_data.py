#%%
from pathlib import Path, PurePath
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from torchvision import datasets, utils

import cv2

seg_data_dir = PurePath('data', 'segmentation_data')
image_path = Path(Path.joinpath(seg_data_dir, 'images'))
mask_path = Path(Path.joinpath(seg_data_dir, 'masks'))

#%%

# image_files = []
# for img_name in image_path.iterdir():
#     img = plt.imread(img_name.as_posix())
#     image_files.append(img)
# 
# mask_files = []
# for mask_name in mask_path.iterdir():
#     mask = plt.imread(mask_name.as_posix())
#     mask_files.append(mask)

#%%

def show_mask(image, mask):
    plt.imshow(image)
    plt.imshow(mask, cmap='gray', alpha=0.5)

#%%
transform = T.Compose([T.Resize((512, 512)), T.CenterCrop(512), T.ToTensor()])

class SudokuDataset(Dataset):
    """Sudoku Segmentation dataset."""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all images and masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.images = [img for img in image_path.iterdir() if img.is_file()]
        self.masks = [mask for mask in mask_path.iterdir() if mask.is_file()]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = Image.open(self.images[idx].as_posix())
        mask = Image.open(self.masks[idx].as_posix())
        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            sample['mask'] = self.transform(sample['mask'])
        
        return sample


# %%
sudoku_dataset = SudokuDataset(seg_data_dir, transform=transform)

for i in range(len(sudoku_dataset)):
    sample = sudoku_dataset[i]
    print(i, ' ', sample['image'].shape, sample['mask'].shape)

# %%

plt.figure()
show_mask(sample['image'].permute(1, 2, 0).numpy(), sample['mask'].squeeze().numpy())
plt.show()

#%%

img = plt.imread('data/segmentation_data/images/0012_01.jpg')
mask = sample['mask'].squeeze().numpy()
height, width, _ = img.shape
size = min(height, width)
img = img[int((height-size)/2) : int((height - size)/2) + size,
          int((width - size)/2) : int((width - size)/2) + size]
mask = cv2.resize(mask, (size, size))
plt.figure()
show_mask(img, mask)
plt.show()

# %%

sudoku_data_loader = DataLoader(sudoku_dataset, batch_size=4, shuffle=True)

def show_sudoku_batch(sample_batch):
    """Show image with mask for a batch of samples"""
    images_batch, masks_batch = sample_batched['image'], sample_batched['mask']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    image_grid = utils.make_grid(images_batch)
    mask_grid = utils.make_grid(masks_batch)
    mask_grid[mask_grid != 0] = 1
    plt.imshow(image_grid.numpy().transpose(1, 2, 0))
    plt.imshow(mask_grid.numpy().transpose(1, 2, 0), alpha=0.5)

for i_batch, sample_batched in enumerate(sudoku_data_loader):
    print(i_batch, sample_batched['image'].size(), sample_batched['mask'].size())

    plt.figure()
    show_sudoku_batch(sample_batched)
    plt.axis('off')
    plt.ioff()
    plt.show()

# %%
