from torch import nn
import math

class ScaleDotProductAttention(nn.Module):
    """
    attention机制，计算scale dot product attention
    表示decoder的序列与encoder的相关程度
    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """
    def __init__(self) -> None:
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, q, k, v, mask=None, e=1e12):
        # input [batch_size, head, length, d_tensor]
        # k输入时四维，当前计算用到2维，向量维度和seq长度
        batch_size, head, length, d_tensor = k.size()
        # 计算相似度通过Query和K的转置点乘
        k_t = k.view(batch_size,head,d_tensor,length)
        score = (q @ k_t) / math.sqrt(d_tensor)

        # 2. apply masking (opt) , 对较短的句子进行mask操作
        # <pad>位置上对应的内容希望为0，用masked_fill，将所有原<pad>位置的数设置为-inf. 
        if mask is not None:
            mask = mask.unsqueeze(1)
            score = score.masked_fill(mask == 0, -e)
        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)
        # 4. multiply with Value , 权重相乘得到attention向量
        # [L×L].[L×d]->[L×d]
        v = score @ v
        return v,score