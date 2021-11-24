from torch import nn
from models.layers.scale_dot_product_attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        #  3. do scale dot product to compute similarity
        # 对分配好的每个head用attention进行计算
        out, attention = self.attention(q,k,v,mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        # 使用一个线性变换(512,512)，shape不变
        out = self.w_concat(out)

        return out
        
    def split(self, tensor):
        """
            Q,K,V可以看做是每个输入经过8个attention head，每个head的size是8，（8*8=64=embed_size）。
            我们需要将每个head得到的结果分开，然后在每个head中进行softmax的操作
            [batch_size, max_len, 1, d_model] -> [batch_size, n_head, max_len, head_size]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, self.n_head, length, d_tensor)

        return tensor

    def concat(self, tensor):
        """
            将划分的多个head利用attention的结果进行拼接
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.view(batch_size, length, d_model)
        return tensor