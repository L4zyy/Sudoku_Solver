from pathlib import Path
import numpy as np
from cv2 import cv2 as cv
import matplotlib.pyplot as plt
from time import time
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms, datasets


class sdkSolver():
    def __init__(self, seg_model_path, ocr_model_path):
        self.seg_model = torch.load(seg_model_path)
        self.ocr_model = torch.load(ocr_model_path)

        self.current_image = None
        self.current_image = None
    
    def solve(self):
        pass

    def print(self):
        pass