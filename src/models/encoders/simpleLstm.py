import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.utils.gpu_tools import move2cuda

class SimpleLSTM(nn.Module):
    def __init__(self, layerUtils):
        super(SimpleLSTM, self).__init__()
        self.rnn, self.initHiddenWithBatchSize = layerUtils.getEncoderRNN()
        self.dropout = layerUtils.getDropOut()

    def forward(self, inputs, inputSeqLengths):
        sentLength = inputs.shape[1]
        sortedSeqLengths, permIdx = inputSeqLengths.sort(0, descending=True)
        _, recoverPermIdx = permIdx.sort(0, descending=False)
        sortedSeqTensors = inputs[permIdx]

        packedWords = pack_padded_sequence(sortedSeqTensors, sortedSeqLengths, True)
        hidden = self.initHiddenWithBatchSize(len(inputs))
        lstmOut, hidden = self.rnn(packedWords, hidden)
        lstmOut, _ = pad_packed_sequence(lstmOut, batch_first=True)
        if lstmOut.shape[1] < sentLength:
            pad = torch.zeros(inputs.shape[0], sentLength-lstmOut.shape[1], lstmOut.shape[2])
            pad = move2cuda(pad)
            return torch.cat([lstmOut[recoverPermIdx], pad], 1)
        else: 
            return lstmOut[recoverPermIdx]
