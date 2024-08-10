import time

import torch
from torch import nn
from torchmetrics.regression import PearsonCorrCoef


class Trainer:
    def __init__(self, options, model, device, train_loader, optimizer, test_loader):
        self.train_loader = train_loader
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.test_loader = test_loader
        self.dry_run = options.dry_run
        self.clip_norm = options.clip_norm
        self.model_path = options.model_path
        self.log_interval = options.log_interval
        self.epoch = 0
        self.epoch_time = 0
        
        self.logs = ""
        
        # early stop
        self.patience = options.patience
        self.epochs_since_improvement = 0
        self.min_test_loss = 1
    
    @staticmethod
    def pearson_correlation_coefficient(a, b):
        r = torch.zeros(a.size(0), 3)
        for i in range(a.size(0)):
            for j in range(3):
                r[i, j] = PearsonCorrCoef()(a[i, :, j], b[i, :, j])
        return r
    
    def train(self, epoch):
        self.model.train()
        start_time = time.time()
        t_loss_list = []
        t_pearsonr_list = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            loss_list = []
            pearsonr_list = []
            data, target = data.to(self.device), target
            self.optimizer.zero_grad()
            output = self.model(data).to('cpu')
            loss = output - target * torch.log(output)
            loss = torch.mean(loss, dim=(1, 2)) + self.model.l2_regularization(0.001).to('cpu')
            loss_list.extend(loss.tolist())
            pearsonr_list.extend(self.pearson_correlation_coefficient(output, target).mean(dim=1).tolist())
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm)
            self.optimizer.step()
            if batch_idx != 0 and batch_idx % self.log_interval == 0:
                print('Train Epoch: {:3} [{:4}/{:4} ({:3.0f}%)] \tLoss: {:.4f}, Pearsonr: {:.4f}'.format(
                    self.epoch,
                    self.train_loader.batch_size * batch_idx,
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    sum(loss_list) / len(loss_list),
                    sum(pearsonr_list) / len(pearsonr_list)))
                if self.dry_run:
                    break
            t_loss_list.extend(loss.tolist())
            t_pearsonr_list.extend(pearsonr_list)
        
        t_loss = sum(t_loss_list) / len(t_loss_list)
        t_pearsonr = sum(t_pearsonr_list) / len(t_pearsonr_list)
        self.logs = 'Epoch: {:2} - {:}s - train_loss: {:.4f} - train_r: {:.4f} - '.format(epoch,
                                                                                          round(
                                                                                              time.time() - start_time),
                                                                                          t_loss,
                                                                                          t_pearsonr)
    
    def valid(self):
        self.model.eval()
        t_loss_list = []
        t_pearsonr_list = []
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target
                output = self.model(data).to('cpu')
                loss = output - target * torch.log(output)
                loss = torch.mean(loss, dim=(1, 2))
                t_loss_list.extend(loss.tolist())
                t_pearsonr_list.extend(self.pearson_correlation_coefficient(output, target).mean(dim=1).tolist())
        
        t_loss = sum(t_loss_list) / len(t_loss_list)
        t_pearsonr = sum(t_pearsonr_list) / len(t_pearsonr_list)
        self.logs += 'valid_loss: {:.4f} - valid_r: {:.4f}'.format(t_loss, t_pearsonr)
        # early stop
        if t_loss < self.min_test_loss:
            self.logs += ' - best!'
            self.min_test_loss = t_loss
            self.epochs_since_improvement = 0
            torch.save(self.model.state_dict(), self.model_path)
        else:
            self.epochs_since_improvement += 1
        
        print(self.logs)
        
        if self.epochs_since_improvement >= self.patience:
            print(f'Stopping training early')
            exit(0)
