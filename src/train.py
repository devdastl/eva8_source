#custom module to run training loop

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

class TrainModel():
  def __init__(self, model, device, dataloader, criterion, optimizer, scheduler):
    self.model = model
    self.device = device
    self.dataloader = dataloader
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.criterion = criterion
    self.lr_trend = []
    self.losses = []
    self.acc = []

  def train_a(self, l1_lambda=0.001, L1_reg=False):
      self.model.train()
      pbar = tqdm(self.dataloader)
      correct = 0
      processed = 0
      train_loss = 0

      for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(self.device), target.to(self.device)

        # Init
        self.optimizer.zero_grad()

        # Predict
        y_pred = self.model(data)

        # Calculate loss
        loss = self.criterion(y_pred, target) if not (L1_reg) else (self.criterion(y_pred, target) + self.l1_loss(l1_lambda))
        train_loss += loss

        # Backpropagation
        loss.backward()
        self.optimizer.step()
        # updating LR
        if self.scheduler:
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
                self.lr_trend.append(self.scheduler.get_last_lr()[0])


        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

      self.acc.append(100*correct/processed)
      train_loss /= len(self.dataloader.dataset)
      self.losses.append(train_loss.item())

  # function to implement L1 regularization
  def l1_loss(self, l1_lambda):

    l1_reg = None
    for param in self.model.parameters():
      if l1_reg is None:
        l1_reg = param.norm(1)
      else:
        l1_reg = l1_reg + param.norm(1)
    return l1_reg*l1_lambda