from torch import nn 

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embedding import TokenEmbedding

class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    """
    def __init__(self, vocab_size, d_model, max_len, device, drop_prob) :
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)
    
    def forward(self, x):
        """
        x 表示当前的句子中一个词token
        """
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)
