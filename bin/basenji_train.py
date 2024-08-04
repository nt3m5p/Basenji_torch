from __future__ import print_function

import glob
import os
import warnings
from optparse import OptionParser

import numpy as np
import torch
from natsort import natsorted
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from basenji import seqnn
from basenji.CustomDataset import CustomDataset
from basenji.seqnn import SeqNN
from examples.mnist.main import Net
from torchsummary import summary
import torch.nn.init as init
from torchmetrics.regression import PearsonCorrCoef

warnings.filterwarnings('ignore')


def pearson_correlation_coefficient(a, b):
    a = a.view((-1, 3))
    b = b.view((-1, 3))
    pearson = PearsonCorrCoef(num_outputs=3)
    return torch.mean(pearson(a, b))


def train(options, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = []
    pearsonr = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = torch.clamp(output, min=1e-6, max=1e6)
        loss = output - target * torch.log(output)
        loss = loss.mean()
        train_loss.append(loss.item())
        pearsonr.append(pearson_correlation_coefficient(output, target))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=options.clip_norm)
        optimizer.step()
        if batch_idx % options.log_interval == 0:
            print('Train Epoch: {:3} [{:4}/{:4} ({:3.0f}%)] \tLoss: {:.4f}, Pearsonr: {:.4f}'.format(
                epoch,
                batch_idx * options.batch_size,
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                sum(train_loss) / len(train_loss),
                sum(pearsonr) / len(pearsonr)))
            if options.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    pearsonr = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = torch.clamp(output, min=1e-6, max=1e6)
            loss = output - target * torch.log(output)
            loss = loss.mean()
            test_loss += loss.item() * len(data)
            pearsonr += pearson_correlation_coefficient(output, target) * len(data)
    
    test_loss /= len(test_loader.dataset)
    
    print('\nTest set:   AverageLoss: {:.4f}, Pearsonr: {:.4f}\n'.format(
        test_loss / len(test_loader.dataset), pearsonr / len(test_loader.dataset)))


def main():
    usage = 'usage: %prog [options] <params_file> <data1_dir> ...'
    parser = OptionParser(usage)
    parser.add_option('-m', dest='mixed_precision',
                      default=False, action='store_true',
                      help='Train with mixed precision [Default: %default]')
    parser.add_option('-o', dest='out_dir',
                      default='train_out',
                      help='Output directory for test statistics [Default: %default]')
    parser.add_option('--restore', dest='restore',
                      help='Restore model and continue training [Default: %default]')
    parser.add_option('--trunk', dest='trunk',
                      default=False, action='store_true',
                      help='Restore only model trunk [Default: %default]')
    parser.add_option('--lr', dest='lr', default=0.1, )
    parser.add_option('--gamma', dest='gamma', type=float, default=0.99, )
    parser.add_option('--epochs', dest='epochs', type=int, default=100, )
    parser.add_option('--log_interval', dest='log_interval', type=int, default=25, )
    parser.add_option('--dry_run', dest='dry_run', default=False)
    parser.add_option('--batch_size', dest='batch_size', default=4)
    parser.add_option('--momentum', dest='momentum', default=0.99)
    parser.add_option('--patience', dest='patience', default=8)
    parser.add_option('--clip_norm', dest='clip_norm', default=2)
    
    (options, args) = parser.parse_args()
    
    data_dirs = args[0:]
    
    device = torch.device("cpu")
    
    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)
    
    # read datasets
    train_data = {"sequence": [], "target": []}
    test_data = {"sequence": [], "target": []}
    for data_dir in data_dirs:
        train_pt_path = '%s/ptrecords/%s-*.pt' % (data_dir, 'train')
        for f in natsorted(glob.glob(train_pt_path)):
            data = torch.load(f)
            train_data["sequence"].extend(data["sequence"])
            train_data["target"].extend(data["target"])
        test_pt_path = '%s/ptrecords/%s-*.pt' % (data_dir, 'test')
        for f in natsorted(glob.glob(test_pt_path)):
            data = torch.load(f)
            test_data["sequence"].extend(data["sequence"])
            test_data["target"].extend(data["target"])
    
    train_dataset = CustomDataset(train_data)
    train_loader = DataLoader(train_dataset, options.batch_size, shuffle=True)
    test_dataset = CustomDataset(test_data)
    test_loader = DataLoader(test_dataset, options.batch_size)
    
    model = SeqNN().to(device)
    for name, param in model.named_parameters():
        if 'weight' in name:
            try:
                init.xavier_normal_(param, gain=1.0)
            except:
                init.normal_(param, mean=0.0, std=1.0)
        else:
            init.constant_(param, 0)
    optimizer = optim.SGD(model.parameters(), lr=options.lr, momentum=options.momentum)
    
    scheduler = StepLR(optimizer, step_size=1, gamma=options.gamma)
    for epoch in range(1, options.epochs + 1):
        train(options, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()


if __name__ == '__main__':
    main()
