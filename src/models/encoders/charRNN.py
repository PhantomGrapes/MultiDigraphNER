import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CharRNN(nn.Module):
    def __init__(self, layerUtil):
        super(CharRNN, self).__init__()
        self.embedding = layerUtil.getCharEmbeddingParameter()
        self.rnn = layerUtil.getCharRNN()
        self.dropout = layerUtil.getDropOut()

    def forward(self, seqTensors, seqLengths):
        batchSize = seqTensors.shape[0]
        sentLength = seqTensors.shape[1]
        seqTensors = seqTensors.view(batchSize * sentLength, -1)
        seqLengths = seqLengths.view(batchSize * sentLength)
        sortedSeqLengths, permIdx = seqLengths.sort(0, descending=True)
        _, recoverPermIdx = permIdx.sort(0, descending=False)
        sortedSeqTensors = seqTensors[permIdx]
        sortedSeqEmbedding = self.dropout(self.embedding(sortedSeqTensors))

        packedChars = pack_padded_sequence(sortedSeqEmbedding, sortedSeqLengths, True)
        lstmOut, _ = self.rnn(packedChars, None)
        lstmOut, _ = pad_packed_sequence(lstmOut, batch_first=True)

        sortedSeqRNNLastHidden = torch.gather(lstmOut, 1, (sortedSeqLengths - 1).view(batchSize * sentLength, 1, 1).expand(batchSize * sentLength, 1, lstmOut.shape[-1]))

        return sortedSeqRNNLastHidden[recoverPermIdx].view(batchSize, sentLength, -1)
