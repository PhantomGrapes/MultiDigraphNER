import torch
from torch import nn
from src.utils.gpu_tools import move2cuda

class LinearChainCRF(nn.Module):
    def __init__(self, layerUtils):
        super(LinearChainCRF, self).__init__()
        self.transitions = layerUtils.getTransitionMatrix()
        self.tagSize = layerUtils.getTagSize()
        self.tagEndId = layerUtils.getTagEndId()
        self.tagStartId = layerUtils.getTagStartId()
        self.useGpu = layerUtils.getUseGpu()

    def forward(self, features, seqLens, mask):
        """
        :param features:  [batchSize * sentLength * tagSize]
        :return: scalar
        """
        batchSize = len(features)
        scores = self.transitions.view(1, 1, self.tagSize, self.tagSize).expand(
            batchSize, features.shape[1], self.tagSize, self.tagSize) + \
                 features.view(batchSize, features.shape[1], self.tagSize, 1).expand(batchSize, features.shape[1], self.tagSize, self.tagSize)
        alpha = torch.zeros([batchSize, features.shape[1], self.tagSize])
        if self.useGpu:
            alpha = move2cuda(alpha)
        alpha[:, 0, :] = scores[:, 0, :, self.tagStartId]
        for wordIdx in range(1, features.shape[1]):
            scoresIdx = alpha[:, wordIdx - 1, :].view(batchSize, 1, self.tagSize).expand(batchSize, self.tagSize, self.tagSize) + scores[:, wordIdx, :, :]
            alpha[:, wordIdx, :] = torch.logsumexp(scoresIdx, 2)

        lastAlpha = torch.gather(alpha, 1, seqLens.view(batchSize, 1, 1).expand(batchSize, 1, self.tagSize) - 1).view(batchSize, self.tagSize)
        lastAlpha = lastAlpha + self.transitions[self.tagEndId].view(1, self.tagSize).expand(batchSize, self.tagSize)
        lastAlpha = torch.logsumexp(lastAlpha, 1).view(batchSize)

        return torch.sum(lastAlpha), scores

    def scoreSentence(self, tagSeqTensor, wordSeqLengths, scores, mask):
        batchSize = scores.shape[0]
        sentLength = scores.shape[1]

        currentTagScores = torch.gather(scores, 2, tagSeqTensor.view(batchSize, sentLength, 1, 1).expand(batchSize, sentLength, 1, self.tagSize)).view(batchSize, -1, self.tagSize)
        tagTransScoresMiddle = torch.gather(currentTagScores[:, 1:, :], 2, tagSeqTensor[:, : sentLength - 1].view(batchSize, sentLength - 1, 1)).view(batchSize, -1)
        tagTransScoresBegin = currentTagScores[:, 0, self.tagStartId]
        endTagIds = torch.gather(tagSeqTensor, 1, wordSeqLengths.view(batchSize, 1) - 1)
        tagTransScoresEnd = torch.gather(self.transitions[self.tagEndId, :].view(1, self.tagSize).expand(batchSize, self.tagSize), 1, endTagIds).view(batchSize)

        return torch.sum(tagTransScoresBegin) + torch.sum(tagTransScoresMiddle.masked_select(mask[:, 1:])) + torch.sum(tagTransScoresEnd)

    def viterbiDecode(self, features, seqLens, scores=None):
        batchSize = len(features)
        scoresRecord = torch.zeros([batchSize, features.shape[1], self.tagSize])
        idxRecord = torch.zeros([batchSize, features.shape[1], self.tagSize], dtype=torch.int64)
        mask = torch.ones_like(seqLens, dtype=torch.int64)
        startIds = torch.full((batchSize, self.tagSize), self.tagStartId, dtype=torch.int64)
        decodeIdx = torch.LongTensor(batchSize, features.shape[1])
        if self.useGpu:
            scoresRecord = move2cuda(scoresRecord)
            idxRecord = move2cuda(idxRecord)
            mask = move2cuda(mask)
            decodeIdx = move2cuda(decodeIdx)
            startIds = move2cuda(startIds)

        if scores is None:
            scores = self.transitions.view(1, 1, self.tagSize, self.tagSize).expand(
                batchSize, features.shape[1], self.tagSize, self.tagSize) + \
                     features.view(batchSize, features.shape[1], self.tagSize, 1).expand(batchSize, features.shape[1],
                                                                                         self.tagSize, self.tagSize)
        # scoresRecord[:, 0, :] = self.getInitAlphaWithBatchSize(batchSize).view(batchSize, self.tagSize)
        scoresRecord[:, 0, :] = scores[:, 0, :, self.tagStartId]
        idxRecord[:,  0, :] = startIds
        for wordIdx in range(1, features.shape[1]):
            scoresIdx = scoresRecord[:, wordIdx - 1, :].view(batchSize, 1, self.tagSize).expand(batchSize, self.tagSize,
                                                                                  self.tagSize) + scores[:, wordIdx, :, :]
            idxRecord[:, wordIdx, :] = torch.argmax(scoresIdx, 2)
            scoresRecord[:, wordIdx, :] = torch.gather(scoresIdx, 2, idxRecord[:, wordIdx, :].view(batchSize, self.tagSize, 1)).view(batchSize, self.tagSize)

        lastScores = torch.gather(scoresRecord, 1, seqLens.view(batchSize, 1, 1).expand(batchSize, 1, self.tagSize) - 1).view(batchSize, self.tagSize)
        lastScores = lastScores + self.transitions[self.tagEndId].view(1, self.tagSize).expand(batchSize, self.tagSize)
        decodeIdx[:, 0] = torch.argmax(lastScores, 1)
        bestScores = torch.gather(lastScores, 1, decodeIdx[:, 0].view(batchSize, 1))

        for distance2Last in range(features.shape[1] - 1):
            lastNIdxRecord = torch.gather(idxRecord, 1, torch.where(seqLens - distance2Last - 1 > 0, seqLens - distance2Last - 1, mask).view(batchSize, 1, 1).expand(batchSize, 1, self.tagSize)).view(batchSize, self.tagSize)
            decodeIdx[:, distance2Last + 1] = torch.gather(lastNIdxRecord, 1, decodeIdx[:, distance2Last].view(batchSize, 1)).view(batchSize)

        return bestScores, decodeIdx


