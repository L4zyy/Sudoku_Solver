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

def show_mask(image, mask):
    plt.imshow(image)
    plt.imshow(mask, cmap='gray', alpha=0.5)

class SudokuDataset(Dataset):
    """Sudoku Segmentation dataset."""
    
    def __init__(self, seg_data_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all images and masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.seg_data_dir = seg_data_dir
        self.transform = transform

        image_path = Path(Path.joinpath(seg_data_dir, 'images'))
        mask_path = Path(Path.joinpath(seg_data_dir, 'masks'))

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
        
        sample['mask'][sample['mask'] != 0] = 1
        
        return sample

def show_sudoku_batch(sample_batched):
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

if __name__ == "__main__":
    seg_data_dir = PurePath('data', 'segmentation_data')

    transform = T.Compose([T.Resize((512, 512)), T.CenterCrop(512), T.ToTensor()])

    sudoku_dataset = SudokuDataset(seg_data_dir, transform=transform)

    train_size =int(0.8*len(sudoku_dataset))
    val_size = len(sudoku_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(sudoku_dataset, [train_size, val_size])

    print(len(train_dataset), ' ', len(val_dataset))

    sudoku_data_loader = DataLoader(val_dataset, batch_size=3, shuffle=True)

    for i_batch, sample_batched in enumerate(sudoku_data_loader):
        print(i_batch, sample_batched['image'].size(), sample_batched['mask'].size())

        plt.figure()
        show_sudoku_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()