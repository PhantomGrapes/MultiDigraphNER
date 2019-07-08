import torch
from torch import nn

class SimpleEmbedding(nn.Module):
    def __init__(self, layerUtil):
        super(SimpleEmbedding, self).__init__()
        self.embedding = layerUtil.getEmbeddingParameter()
        self.dropout = layerUtil.getDropOut()
        self.wordEmbedding = None
        self.eit = None

    def forward(self, seqTensor):
        self.wordEmbedding = self.embedding(seqTensor)
        return self.wordEmbedding