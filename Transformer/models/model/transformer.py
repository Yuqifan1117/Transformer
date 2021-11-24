import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder
import torch.nn.functional as F

class Generator(nn.Module):
    """
    解码器输出经线性变换和softmax函数映射为下一时刻预测单词的概率分布
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # decode后的结果，先进入一个全连接层变为词典大小的向量
        self.proj = nn.Linear(vocab, d_model)

    def forward(self, x):
        # 然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
        return F.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):

    def __init__(self, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, generator, device):
        super(Transformer, self).__init__()
        self.device = device
        self.encoder = Encoder(d_model=d_model,  ## emb_size
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_vocab_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_vocab_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
        self.generator = generator.to(device)

    def forward(self, src, trg, src_mask, trg_mask):
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    # def make_pad_mask(self, q, k):
    #     len_q, len_k = q.size(1), k.size(1)

    #     # batch_size x 1 x 1 x len_k
    #     k = k.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    #     # batch_size x 1 x len_q x len_k
    #     k = k.repeat(1, 1, len_q, 1)

    #     # batch_size x 1 x len_q x 1
    #     q = q.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(3)
    #     # batch_size x 1 x len_q x len_k
    #     q = q.repeat(1, 1, 1, len_k)

    #     mask = k & q
    #     return mask

    # def make_no_peak_mask(self, q, k):
    #     len_q, len_k = q.size(1), k.size(1)

    #     # len_q x len_k
    #     mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)

    #     return mask