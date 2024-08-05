import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticShift(nn.Module):
    def __init__(self, max_shift):
        super(StochasticShift, self).__init__()
        self.max_shift = max_shift
    
    def forward(self, x):
        shift_amount = torch.randint(-self.max_shift, self.max_shift + 1, (1,))
        return torch.roll(x, shifts=shift_amount.item(), dims=-1)


class StochasticReverseComplement(nn.Module):
    def __init__(self, reverse):
        super(StochasticReverseComplement, self).__init__()
        self.reverse = reverse
    
    def forward(self, x):
        return x


class SwitchReverse(nn.Module):
    def __init__(self, reverse):
        super(SwitchReverse, self).__init__()
        self.reverse = reverse
    
    def forward(self, x):
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels, momentum=0.9)
    
    def forward(self, x):
        x = F.gelu(x)
        x = self.conv(x)
        return self.bn(x)


class ConvConvDropBlock(nn.Module):
    def __init__(self, cov1=None, cov2=None, dropout=0.25, dilation=1):
        super(ConvConvDropBlock, self).__init__()
        if cov1 is None:
            cov1 = [72, 32, 3, 1, 1, dilation]
        if cov2 is None:
            cov2 = [32, 72, 1, 1, 0, 1]
        self.conv_block1 = ConvBlock(cov1[0], cov1[1], cov1[2], cov1[3], cov1[4])
        self.conv_block2 = ConvBlock(cov2[0], cov2[1], cov2[2], cov2[3], cov2[4])
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return self.dropout(x)


class SeqNN(nn.Module):
    def __init__(self):
        super(SeqNN, self).__init__()
        self.reverse = True
        self.stochastic_reverse_complement = StochasticReverseComplement(self.reverse)
        self.stochastic_shift = StochasticShift(3)
        self.conv_block = ConvBlock(4, 64, kernel_size=15, stride=1, padding=7)
        self.conv_block1 = ConvBlock(64, 64, kernel_size=5, stride=1, padding=2)
        self.conv_block2 = ConvBlock(64, 72, kernel_size=5, stride=1, padding=2)
        self.conv_conv_dropout = ConvConvDropBlock(dilation=1)
        self.conv_conv_dropout1 = ConvConvDropBlock(dilation=2)
        self.conv_conv_dropout2 = ConvConvDropBlock(dilation=4)
        self.conv_conv_dropout3 = ConvConvDropBlock(dilation=8)
        self.conv_conv_dropout4 = ConvConvDropBlock(dilation=16)
        self.conv_conv_dropout5 = ConvConvDropBlock(dilation=32)
        self.conv_block3 = ConvBlock(72, 64, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(p=0.05)
        self.fc1 = nn.Linear(64, 3)
        self.switch_reverse = SwitchReverse(self.reverse)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        stochastic_reverse_complement = self.stochastic_reverse_complement(x)
        stochastic_shift = self.stochastic_shift(stochastic_reverse_complement)
        batch_normalization = self.conv_block(stochastic_shift)
        max_pooling1d = F.max_pool1d(batch_normalization, kernel_size=8)
        batch_normalization_1 = self.conv_block1(max_pooling1d)
        max_pooling1d_1 = F.max_pool1d(batch_normalization_1, kernel_size=4)
        batch_normalization_2 = self.conv_block2(max_pooling1d_1)
        max_pooling1d_2 = F.max_pool1d(batch_normalization_2, kernel_size=4)
        
        dropout = self.conv_conv_dropout(max_pooling1d_2)
        add = max_pooling1d_2 + dropout
        dropout_1 = self.conv_conv_dropout1(add)
        add_1 = add + dropout_1
        dropout_2 = self.conv_conv_dropout2(add_1)
        add_2 = add_1 + dropout_2
        dropout_3 = self.conv_conv_dropout3(add_2)
        add_3 = add_2 + dropout_3
        dropout_4 = self.conv_conv_dropout4(add_3)
        add_4 = add_3 + dropout_4
        dropout_5 = self.conv_conv_dropout5(add_4)
        add_5 = add_4 + dropout_5
        batch_normalization = self.conv_block3(add_5)
        dropout_6 = self.dropout(batch_normalization)
        gelu_16 = F.gelu(dropout_6)
        gelu_16 = gelu_16.permute(0, 1, 2).view(-1, 64)
        dense = self.fc1(gelu_16)
        dense = dense.view(-1, 1024, 3)
        switch_reverse = self.switch_reverse(dense)
        self.reverse = not self.reverse
        return F.softplus(switch_reverse)
