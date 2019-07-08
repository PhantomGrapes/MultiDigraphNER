import torch
from torch import nn
from src.models.encoders.charRNN import CharRNN

class CharWordEmbedding(nn.Module):
    def __init__(self, layerUtil):
        super(CharWordEmbedding, self).__init__()
        self.embedding = layerUtil.getEmbeddingParameter()
        self.charEmbedding = CharRNN(layerUtil)
        self.dropout = layerUtil.getDropOut()
        self.wordEmbedding = None
        self.eit = None

    def forward(self, wordSeqTensors, charSeqTensors, charSeqLengths):
        self.wordEmbedding = self.embedding(wordSeqTensors)
        charRNNEmbedding = self.charEmbedding(charSeqTensors, charSeqLengths)
        mergeEmbedding = torch.cat([self.wordEmbedding, charRNNEmbedding], 2)
        wordEmbeddingDropOut = self.dropout(mergeEmbedding)
        return wordEmbeddingDropOut
