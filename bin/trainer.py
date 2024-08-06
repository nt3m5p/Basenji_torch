import torch
from torch import nn
from scipy.stats import spearmanr, pearsonr


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
        
        # early stop
        self.patience = options.patience
        self.epochs_since_improvement = 0
        self.min_test_loss = 1
    
    @staticmethod
    def pearson_correlation_coefficient(a, b):
        a = a.view((4, -1)).detach().numpy()
        b = b.view((4, -1)).detach().numpy()
        r = [pearsonr(a[i], b[i]).statistic for i in range(4)]
        return r
    
    def train(self, epoch):
        self.model.train()
        train_loss = []
        pearsonr_list = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            pearsonr_list.extend(self.pearson_correlation_coefficient(output, target))
            loss = output - target * torch.log(output)
            loss = torch.mean(loss, dim=(1, 2))
            train_loss.extend(loss.tolist())
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm)
            self.optimizer.step()
            if batch_idx != 0 and batch_idx % self.log_interval == 0:
                print('Train Epoch: {:3} [{:4}/{:4} ({:3.0f}%)] \tLoss: {:.4f}, Pearsonr: {:.4f}'.format(
                    epoch,
                    self.train_loader.batch_size * batch_idx,
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    sum(train_loss) / len(train_loss),
                    sum(pearsonr_list) / len(pearsonr_list)))
                if self.dry_run:
                    break
            train_loss = []
            pearsonr_list = []
    
    def test(self):
        self.model.eval()
        test_loss = 0
        pearsonr_list = []
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pearsonr_list.extend(self.pearson_correlation_coefficient(output, target))
                loss = output - target * torch.log(output)
                loss = loss.mean()
                test_loss += loss.item() * len(data)
        
        test_loss /= len(self.test_loader.dataset)
        pearsonr_list = sum(pearsonr_list) / len(pearsonr_list)
        
        # early stop
        if test_loss < self.min_test_loss:
            print('\nTest set:    AverageLoss: {:.4f}, Pearsonr: {:.4f}  best!\n'.format(test_loss, pearsonr_list))
            self.min_test_loss = test_loss
            self.epochs_since_improvement = 0
            torch.save(self.model.state_dict(), self.model_path)
        else:
            print('\nTest set:   AverageLoss: {:.4f}, Pearsonr: {:.4f}\n'.format(test_loss, pearsonr_list))
            self.epochs_since_improvement += 1
        
        if self.epochs_since_improvement >= self.patience:
            print(f'Stopping training early')
            exit(0)
