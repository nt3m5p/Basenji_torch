import torch
from torch import nn
from torchmetrics import PearsonCorrCoef


class Trainer:
    def __init__(self, options, model, device, train_loader, optimizer, test_loader):
        self.train_loader = train_loader
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.test_loader = test_loader
        self.dry_run = options.dry_run
        self.clip_norm = options.clip_norm
        self.log_interval = options.log_interval
        self.model_path = options.model_path
        
        # early stop
        self.patience = options.patience
        self.epochs_since_improvement = 0
        self.min_test_loss = 1
        
    @staticmethod
    def pearson_correlation_coefficient(a, b):
        a = a.view((-1))
        b = b.view((-1))
        pearson = PearsonCorrCoef()
        return pearson(a, b)
    
    def train(self, epoch):
        train_loss = []
        pearsonr = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            output = torch.clamp(output, min=1e-6, max=1e6)
            loss = output - target * torch.log(output)
            loss = torch.mean(loss, dim=(1, 2))
            train_loss.extend(loss.tolist())
            loss.mean().backward()
            pearsonr.append(self.pearson_correlation_coefficient(output, target) * len(data))
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm)
            self.optimizer.step()
            
            if batch_idx != 0 and batch_idx % self.log_interval == 0:
                print('Train Epoch: {:3} [{:4}/{:4} ({:3.0f}%)] \tLoss: {:.4f}, Pearsonr: {:.4f}'.format(
                    epoch,
                    self.train_loader.batch_size*batch_idx,
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    sum(train_loss) / len(train_loss),
                    sum(pearsonr) / len(pearsonr)))
                train_loss = []
                pearsonr = []
                if self.dry_run:
                    break
    
    def test(self):
        self.model.eval()
        test_loss = 0
        pearsonr = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                output = torch.clamp(output, min=1e-6, max=1e6)
                loss = output - target * torch.log(output)
                loss = loss.mean()
                test_loss += loss.item() * len(data)
                pearsonr += self.pearson_correlation_coefficient(output, target) * len(data)
        
        test_loss /= len(self.test_loader.dataset)
        pearsonr /= len(self.test_loader.dataset)
        
        # early stop
        if test_loss < self.min_test_loss:
            print('\nTest set:    AverageLoss: {:.4f}, Pearsonr: {:.4f}  best!\n'.format(test_loss, pearsonr))
            self.min_test_loss = test_loss
            self.epochs_since_improvement = 0
            torch.save(self.model.state_dict(), self.model_path)
        else:
            print('\nTest set:   AverageLoss: {:.4f}, Pearsonr: {:.4f}\n'.format(test_loss, pearsonr))
            self.epochs_since_improvement += 1

        if self.epochs_since_improvement >= self.patience:
            print(f'Stopping training early')
            exit(0)