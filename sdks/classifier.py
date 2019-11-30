import torch
import torch.nn.functional as F
from torch.autograd import Variable

from collections import OrderedDict

from tqdm import tqdm
import helpers
import tools
import losses as losses_utils

class SudokuClassifier:
    def __init__(self, net, max_epochs):
        self.net = net
        self.max_epochs = max_epochs
        self.epoch_count = 0
        self.use_cuda = torch.cuda.is_available()
    
    def restore_model(self, path):
        self.net.load_state_dict(torch.load(path))

    def _criterion(self, logits, labels):
        return losses_utils.BinaryCrossEntropyLoss().forward(logits, labels) + losses_utils.SoftDiceLoss().forward(logits, labels)

    def _val_epoch(self, val_loader, threshold):
        losses = tools.AverageMeter()
        dice_coeffs = tools.AverageMeter()

        batch_size = val_loader.batch_size
        it_num = len(val_loader)

        with tqdm(total=it_num, desc="Validating", leave=False) as pbar:
            for idx, sample in enumerate(val_loader):
                inputs = sample['image']
                targets = sample['mask']
                if self.use_cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                
                # images = Variable(images, volatile=True)
                # targets = Variable(targets, volatile=True)

                # forward
                logits = self.net(inputs)
                probs = F.sigmoid(logits)
                preds = (probs > threshold).float()

                loss = self._criterion(logits, targets)
                acc = losses_utils.dice_coeff(preds, targets)
                losses.update(loss, batch_size)
                dice_coeffs.update(acc, batch_size)
                pbar.update(1)
        
        return losses.avg, dice_coeffs.avg, acc

    def _train_epoch(self, train_loader, optimizer, threshold):
        losses = tools.AverageMeter()
        dice_coeffs = tools.AverageMeter()

        batch_size = train_loader.batch_size
        it_num = len(train_loader)

        with tqdm(
            total=it_num,
            desc="Epochs {}/{}".format(self.epoch_count + 1, self.max_epochs),
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]'
        ) as pbar:
            for idx, sample in enumerate(train_loader):
                inputs = sample['image']
                targets = sample['mask']
                if self.use_cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)

                # forward
                logits = self.net.forward(inputs)
                probs = F.sigmoid(logits)
                pred = (probs > threshold).float()

                # back prop + optimize
                loss = self._criterion(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print statistics
                acc = losses_utils.dice_coeff(pred, targets)

                losses.update(loss, batch_size)
                dice_coeffs.update(acc, batch_size)

                # update pbar
                pbar.set_postfix(OrderedDict(loss='{0:1.5f}'.format(loss),
                                             dice_coeff='{0:1.5f}'.format(acc)))
                pbar.update(1)
        return losses.avg, dice_coeffs.avg

    @helpers.timer
    def _run_epoch(self, train_loader, val_loader, optimizer, threshold=0.5):
        self.net.train()
        train_loss, train_dice_coeff = self._train_epoch(train_loader, optimizer, threshold)

        self.net.eval()
        val_loss, val_dice_coeff, acc = self._val_epoch(val_loader, threshold)

        print("train_loss = {:03f}, train_dice_coeff = {:03f}\n"
              "val_loss   = {:03f}, val_dice_coeff   = {:03f}\n"
              "val_acc   = {:03f}"
              .format(train_loss, train_dice_coeff, val_loss, val_dice_coeff, acc))
        self.epoch_count += 1

    def train(self, train_loader, val_loader, optimizer, epochs, threshold=0.5):
        if self.use_cuda:
            self.net.cuda()
        
        for epoch in range(epochs):
            self._run_epoch(train_loader, val_loader, optimizer, threshold)

    def predict(self, test_loader):
        pass