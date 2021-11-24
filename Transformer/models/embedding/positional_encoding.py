import torch
from torch import nn

class PositionalEncoding(nn.Module):
    """
    计算position embedding，用sin PE代替
    每个token的position embedding 维度: d_model
    max_len : the length of sentence
    """
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding,self).__init__()
        self.encoding = torch.zeros(max_len,d_model,device=device)
        self.encoding.requires_grad = False # 不需要利用sensor计算梯度

        pos = torch.arange(0, max_len, device=device) # 创建位置index数组
        pos = pos.float().unsqueeze(dim=1)
        
        _2i = torch.arange(0, d_model, step=2, device=device) # 维度i以2i的形式变化

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** _2i / d_model))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** _2i / d_model))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]
        # [batch_size,max_len,1,embed_size]   
        return self.encoding[:seq_len, :]