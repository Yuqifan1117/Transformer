from torch import nn

class TokenEmbedding(nn.Embedding):
    '''
        vocab_size : size of the dictionary of embeddings
        d_model: the size of each embedding vector
        padding_idx: the embedding vector at :attr:`padding_idx` will default to all zeros,
            句子由若干token组成，对于较短的句子在设置相应的idx表示进行padding
    '''
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)