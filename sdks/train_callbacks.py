import cv2
import torch
import numpy as np
import scipy.misc as scipy
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class Callback:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class TensorboardVisualizerCallback(Callback):
    def __init__(self, path):
        self.path = path
    
    def _apply_mask_overlay(self, image, mask, color=(0, 255, 0)):
        mask = np.dstack((mask, mask, mask)) * np.array(color)
        mask = mask.astype(np.uint8)
        return cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    
    def _get_mask_representation(self, image, mask):
        height, width, channels = image.shape
        results = np.zeros((height, 2*width, 3), np.uint8)

        masked_img = self._apply_mask_overlay(image, mask)

        results[:, 0:width] = image
        results[:, width:2*width] = masked_img

        return results
    
    def __call__(self, *args, **kwargs):
        if kwargs['step_name'] != 'epoch':
            return
        
        epoch_id = kwargs['epoch_id']
        last_images, last_targets, last_preds = kwargs['last_val_batch']
        writer = SummaryWriter(self.path)
        for i, (image, target_mask, pred_mask) in enumerate(zip(last_images, last_targets, last_preds)):
            # transfer float data to uint8
            image = image.data.cpu().numpy()
            image *= 255
            image = image.astype(np.uint8)
            image = np.transpose(image, (1, 2, 0))  # Invert c, h, w to h, w, c

            target_mask = target_mask.float().data.cpu().numpy().astype(np.uint8)
            pred_mask = pred_mask.float().data.cpu().numpy().astype(np.uint8)

            expected_result = self._get_mask_representation(image, target_mask)
            pred_result = self._get_mask_representation(image, pred_mask)
            expected_result = np.transpose(expected_result, (2, 0, 1))
            pred_result = np.transpose(pred_result, (2, 0, 1))

            writer.add_image("Epoch_" + str(epoch_id) + '-Image_' + str(i + 1) + '-Expected', expected_result, epoch_id)
            writer.add_image("Epoch_" + str(epoch_id) + '-Image_' + str(i + 1) + '-Predicted', pred_result, epoch_id)
            if i == 1:  # 2 Images are sufficient
                break
        writer.close()

class TensorboardLoggerCallback(Callback):
    def __init__(self, path):
        self.path = path
    
    def __call__(self, *args, **kwargs):
        if kwargs['step_name'] != 'epoch':
            return
        
        epoch_id = kwargs['epoch_id']

        writer = SummaryWriter(self.path)
        writer.add_scalar('data/train_loss', kwargs['train_loss'], epoch_id)
        writer.add_scalar('data/train_dice_coeff', kwargs['train_dice_coeff'], epoch_id)
        writer.add_scalar('data/val_loss', kwargs['val_loss'], epoch_id)
        writer.add_scalar('data/val_dice_coeff', kwargs['val_dice_coeff'], epoch_id)
        writer.close()