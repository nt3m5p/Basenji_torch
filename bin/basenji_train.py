from __future__ import print_function

import warnings
from optparse import OptionParser

import torch
import torch.nn.init as init
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from basenji.CustomDataset import CustomDataset
from basenji.seqnn import SeqNN
from basenji.trainer import Trainer

warnings.filterwarnings('ignore')


def main():
    usage = 'usage: %prog [options]  <data1_dir> ...'
    parser = OptionParser(usage)
    parser.add_option('--lr', dest='lr', type= float, default=0.1, )
    parser.add_option('--gamma', dest='gamma', type=float, default=0.95, )
    parser.add_option('--epochs', dest='epochs', type=int, default=100, )
    parser.add_option('--dry_run', dest='dry_run', default=False)
    parser.add_option('--batch_size', dest='batch_size', default=4)
    parser.add_option('--momentum', dest='momentum', default=0.99)
    parser.add_option('--patience', dest='patience', default=8)
    parser.add_option('--clip_norm', dest='clip_norm', default=2)
    parser.add_option('--model_path', dest='model_path', default="models/heart/best_model.pth")
    parser.add_option('--restore', dest='restore', default=False)
    parser.add_option('--log_interval', dest='log_interval', default=50, )
    (options, args) = parser.parse_args()
    data_dirs = args[0:]
    
    device = torch.device("cpu")
    
    # read datasets
    train_dataset = CustomDataset(data_dirs, '%s/ptrecords/train-*.pt')
    valid_dataset = CustomDataset(data_dirs, '%s/ptrecords/valid-*.pt')
    train_loader = DataLoader(train_dataset, options.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, options.batch_size)
    
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
    #scheduler = StepLR(optimizer, step_size=1, gamma=options.gamma)
    
    # train
    trainer = Trainer(options, model, device, train_loader, optimizer, valid_loader)
    for epoch in range(1, options.epochs + 1):
        trainer.train(epoch)
        trainer.valid()
        #scheduler.step()


if __name__ == '__main__':
    main()
