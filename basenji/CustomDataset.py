import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): A dictionary containing 'data' and 'targets' lists.
        """
        self.data = data_dict["sequence"]
        self.targets = data_dict["target"]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 返回一个包含数据和目标的元组
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)