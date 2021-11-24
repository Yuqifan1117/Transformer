from torch import nn 

from models.embedding.transformer_embedding import TransformerEmbedding
from models.blocks.encoder_layer import EncoderLayer

class Encoder(nn.Module):
    """
    encoder将初始序列转化成表示向量
    Transformer获取初始序列的embedding传给encoder层
    encoder根据n_layers的数量构建encoder layer
    """
    def __init__(self, enc_vocab_size, d_model, max_len, device, drop_prob, ffn_hidden, n_head, n_layers):
        super().__init__()
        self.emb = TransformerEmbedding(vocab_size=enc_vocab_size, d_model=d_model, max_len=max_len, device=device, drop_prob=drop_prob)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, drop_prob=drop_prob) for _ in range(n_layers)])

    def forward(self, x, s_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, s_mask)

        return x
