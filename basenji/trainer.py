import time

import numpy as np
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
    
    def pearson_r(self,y_pred, y_true):
        true_mean = torch.mean(y_true, dim=1)
        true_mean2 = torch.square(true_mean)
        pred_mean = torch.mean(y_pred, dim=1)
        pred_mean2 = torch.square(pred_mean)
        term1 = torch.sum(torch.multiply(y_true, y_pred),1)
        
        true_sum = torch.sum(y_true, dim=1)
        pred_sum = torch.sum(y_pred, dim=1)
        term2 = -torch.multiply(true_mean, pred_sum)
        term3 = -torch.multiply(pred_mean, true_sum)
        
        count = torch.ones_like(y_true)
        count = torch.sum(count, dim=1)
        term4 = torch.multiply(count, torch.multiply(true_mean, pred_mean))
        
        covariance = term1 + term2 + term3 + term4
        
        true_sumsq = torch.sum(torch.square(y_true), dim=1)
        pred_sumsq = torch.sum(torch.square(y_pred), dim=1)
        true_var = true_sumsq - torch.multiply(count, true_mean2)
        pred_var = pred_sumsq - torch.multiply(count, pred_mean2)
        pred_var = torch.where(torch.greater(pred_var, 1e-12), pred_var, np.inf*torch.ones_like(pred_var))
        tp_var = torch.multiply(torch.sqrt(true_var), torch.sqrt(pred_var))
        correlation = torch.divide(covariance, tp_var)
        return correlation
    
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
            loss = torch.mean(loss, dim=(1, 2))# + self.model.l2_regularization(0.001).to('cpu')
            loss_list.extend(loss.tolist())
            t = self.pearson_r(target, output).mean(dim=1).tolist()
            t2 = self.pearson_correlation_coefficient(output, target).mean(dim=1).tolist()
            pearsonr_list.extend(self.pearson_r(output, target).mean(dim=1).tolist())
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
        with torch.no_grad():
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
                    t_pearsonr_list.extend(self.pearson_r(output, target).mean(dim=1).tolist())
            
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
