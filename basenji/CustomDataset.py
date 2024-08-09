import glob

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_dirs, file_pattern):
        self.data = []
        self.targets = []
        
        for data_dir in data_dirs:
            file_list = glob.glob(file_pattern % data_dir)
            for file_path in file_list:
                data = torch.load(file_path, weights_only=True)
                self.data.extend(data['sequence'])
                self.targets.extend(data['target'])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
