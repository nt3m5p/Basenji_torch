from __future__ import print_function

import glob
import os
import warnings
from optparse import OptionParser

import torch
import torch.nn.init as init
from natsort import natsorted
from torch import optim
from torch.utils.data import DataLoader

from basenji.CustomDataset import CustomDataset
from basenji.seqnn import SeqNN
from bin.trainer import Trainer

warnings.filterwarnings('ignore')


def main():
    usage = 'usage: %prog [options]  <data1_dir> ...'
    parser = OptionParser(usage)
    parser.add_option('--lr', dest='lr', default=0.1, )
    parser.add_option('--gamma', dest='gamma', type=float, default=0.99, )
    parser.add_option('--epochs', dest='epochs', type=int, default=100, )
    parser.add_option('--log_interval', dest='log_interval', type=int, default=25, )
    parser.add_option('--dry_run', dest='dry_run', default=False)
    parser.add_option('--batch_size', dest='batch_size', default=4)
    parser.add_option('--test_batch_size', dest='test_batch_size', default=16)
    parser.add_option('--momentum', dest='momentum', default=0.99)
    parser.add_option('--patience', dest='patience', default=8)
    parser.add_option('--clip_norm', dest='clip_norm', default=2)
    parser.add_option('--model_path', dest='model_path', default="models/heart/best_model.pth")
    parser.add_option('--restore', dest='restore', default=False)
    (options, args) = parser.parse_args()
    data_dirs = args[0:]
    
    device = torch.device("cpu")
    
    # read datasets
    train_dataset = CustomDataset(data_dirs, '%s/ptrecords/train-*.pt')
    test_dataset = CustomDataset(data_dirs,  '%s/ptrecords/test-*.pt')
    train_loader = DataLoader(train_dataset, options.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, options.test_batch_size)
    
    # build model
    model = SeqNN().to(device)
    if options.restore:
        model.load_state_dict(torch.load(options.model_path))
        print("Load best model from %s" % options.model_path)
    else:
        for name, param in model.named_parameters():
            if 'weight' in name:
                try:
                    init.xavier_normal_(param, gain=1.0)
                except:
                    init.normal_(param, mean=0.0, std=1.0)
            else:
                init.constant_(param, 0)
    optimizer = optim.SGD(model.parameters(), lr=options.lr, momentum=options.momentum)
    
    # train
    trainer = Trainer(options, model, device, train_loader, optimizer, test_loader)
    for epoch in range(1, options.epochs + 1):
        trainer.train(epoch)
        trainer.test()


if __name__ == '__main__':
    main()
