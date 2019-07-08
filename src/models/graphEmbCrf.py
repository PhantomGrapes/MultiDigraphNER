import torch
import torch.nn as nn
import torch.autograd as autograd
from tqdm import tqdm
from src.utils.gpu_tools import move2cuda
from src.utils.globalVariable import *

class GraphEmbCrf(nn.Module):
    def __init__(self, Config, layerHelper):
        super(GraphEmbCrf, self).__init__()
        self.wordEmbedding = layerHelper.getWordEmbedding()
        self.dropout = layerHelper.layerUtils.getDropOut()

        self.useBigram = Config.model.use_bigram
        if Config.model.use_bigram:
            self.fwbigramEmbedding, self.bwbigramEmbedding = layerHelper.getBigramEmbedding()

        self.stateDim = layerHelper.layerUtils.getGraphDim()
        self.edgeTypes = layerHelper.layerUtils.getEdgeTypes()
        self.nLayer = Config.model.graph_emb.n_layer
        self.graphEmb = nn.ModuleList([layerHelper.getGraphEmbedding() for i in range(self.nLayer)])
        self.embStateLinear = layerHelper.layerUtils.getEmbStateLinear()
        gaEmbs = layerHelper.layerUtils.getGaEmbs()
        gaLinears = layerHelper.layerUtils.getGaEmb2States()
        self.gaEmb = nn.ModuleList([gaEmbs[i] for i in range(len(gaEmbs))])
        self.gaLinear = nn.ModuleList([gaLinears[i] for i in range(len(gaEmbs))])
        self.gaNum = len(gaEmbs)
        self.mainMethod = Config.model.graph_emb.build_main

        self.encoder = layerHelper.getEncoder()
        self.embFeatureLinear = layerHelper.getEmbFeatureLinear()
        self.crf = layerHelper.getCRF()
        self.useGpu = Config.use_gpu
        self.useChar = Config.model.use_char
        self.useRnn = Config.model.graph_emb.use_rnn
        self.logsoftmax = torch.nn.LogSoftmax(dim=2)
        self.logsoftmax4tag = torch.nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()
        self.tagSize = layerHelper.layerUtils.tagSize

    def prepareTraining(self, corpus, corpusMeta):
        for utt in corpus.utterances:
            utt.mainGraph = utt.buildMainGraph(self.mainMethod)
            utt.mainNode = utt.mainGraph[0]
            utt.totalNode = utt.mainGraph[0]
            for gazIdx in range(self.gaNum):
                ga = corpusMeta.gazetters[gazIdx]
                utt.gazGraph.append(utt.buildGazetterGraph(ga.method, ga.name, utt.totalNode))
                utt.totalNode += utt.gazGraph[-1][0]

    def generateBatchInput(self, corpus, corpusMeta, batchSize):
        if not corpus.prepared:
            self.prepareTraining(corpus, corpusMeta)
        word2Idx = corpusMeta.word2idx
        tag2Idx = corpusMeta.tag2Idx
        char2Idx = corpusMeta.char2Idx
        fwbigram2idx = corpusMeta.fwbigram2idx
        bwbigram2idx = corpusMeta.bwbigram2idx
        # inputBatches = []
        totalSize = len(corpus.utterances)
        for batchId in tqdm(range(totalSize // batchSize), disable=GLOBAL_VARIABLE.DISABLE_TQDM):
            batchUtts = corpus.utterances[batchId * batchSize: (batchId + 1) * batchSize]
            wordSeqLengths = torch.LongTensor(list(map(lambda utt: len(utt.tokens), batchUtts)))
            nodeNums = torch.LongTensor(batchSize)
            maxSeqLength = wordSeqLengths.max()
            if self.useChar:
                charSeqLengths = torch.LongTensor(
                    [list(map(lambda tok: len(tok.chars), utt.tokens)) + [1] * (int(maxSeqLength) - len(utt.tokens)) for utt
                     in batchUtts])
                maxCharLength = charSeqLengths.max()
                charSeqTensor = autograd.Variable(torch.zeros((batchSize, maxSeqLength, maxCharLength))).long()
            else:
                charSeqLengths = None
                maxCharLength = None
                charSeqTensor = None
            wordSeqTensor = autograd.Variable(torch.zeros((batchSize, maxSeqLength))).long()
            if self.useBigram:
                fwbigramTensor = autograd.Variable(torch.zeros([batchSize, maxSeqLength])).long()
                bwbigramTensor = autograd.Variable(torch.zeros([batchSize, maxSeqLength])).long()
            else:
                fwbigramTensor = None
                bwbigramTensor = None
            tagSeqTensor = autograd.Variable(torch.zeros((batchSize, maxSeqLength))).long()
            seq2NodeTensor = autograd.Variable(torch.zeros([batchSize, maxSeqLength, 1])).long()
            node2SeqTensor = autograd.Variable(torch.zeros([batchSize, maxSeqLength, 1])).long()
            gazNodeLengths = []
            gazNode2Idxs = []
            maxTotalNode = max([utt.totalNode for utt in batchUtts])
            maxMainNode = max([utt.mainNode for utt in batchUtts])
            for gazIdx in range(self.gaNum):
                batchMaxNodeLength = max([utt.gazGraph[gazIdx][0] for utt in batchUtts])
                gazNode2Idxs.append(autograd.Variable(torch.zeros([batchSize, batchMaxNodeLength])).long())
                gazNodeLengths.append(autograd.variable(torch.zeros([batchSize])).long())
            if self.gaNum > 0:
                gazBlankState = torch.zeros([batchSize, maxTotalNode - maxMainNode])
            else:
                gazBlankState = None

            mainEdges = []
            for idx in range(batchSize):
                nNode, node2seq, seq2node, edges = batchUtts[idx].mainGraph
                nodeNums[idx] = nNode
                node2SeqTensor[idx, :nNode, 0] = torch.LongTensor(node2seq)
                node2SeqTensor[idx, nNode:, 0] = maxSeqLength - 1
                seq2NodeTensor[idx, :wordSeqLengths[idx], 0] = torch.LongTensor(seq2node)
                mainEdges.append(edges)
            adjMatrixTensor = autograd.Variable(torch.zeros([batchSize, maxTotalNode, maxTotalNode * self.edgeTypes]))

            for idx in range(batchSize):
                mainTypes = len(mainEdges[idx])
                for typeIdx in range(len(mainEdges[idx])):
                    for edge in mainEdges[idx][typeIdx]:
                        adjMatrixTensor[idx, edge[0], maxTotalNode * typeIdx + edge[1]] = edge[2]

                for gazIdx in range(self.gaNum):
                    nNode, node2idx, edges = batchUtts[idx].gazGraph[gazIdx]
                    gazNodeLengths[gazIdx][idx] = nNode
                    gazNode2Idxs[gazIdx][idx][:nNode] = torch.LongTensor(node2idx)
                    for typeIdx in range(len(edges)):
                        for edge in edges[typeIdx]:
                            adjMatrixTensor[idx, edge[0], maxTotalNode * (mainTypes + typeIdx) + edge[1]] = edge[2]

                wordSeqTensor[idx, :wordSeqLengths[idx]] = torch.LongTensor(
                    [word2Idx.get(word.text, corpusMeta.unk) for word in batchUtts[idx].tokens])
                tagSeqTensor[idx, :wordSeqLengths[idx]] = torch.LongTensor(
                    [tag2Idx[word.tag] for word in batchUtts[idx].tokens])
                if self.useBigram:
                    fwbigramTensor[idx, :wordSeqLengths[idx]] = torch.LongTensor([fwbigram2idx.get(word.fwbigram, corpusMeta.unk) for word in batchUtts[idx].tokens])
                    bwbigramTensor[idx, :wordSeqLengths[idx]] = torch.LongTensor([bwbigram2idx.get(word.bwbigram, corpusMeta.unk) for word in batchUtts[idx].tokens])
                if self.useChar:
                    for wordIdx in range(wordSeqLengths[idx]):
                        charSeqTensor[idx, wordIdx, :charSeqLengths[idx, wordIdx]] = torch.LongTensor(
                            [char2Idx.get(char, corpusMeta.unk) for char in batchUtts[idx].tokens[wordIdx].chars])
                    for wordIdx in range(wordSeqLengths[idx], maxSeqLength):
                        charSeqTensor[idx, wordIdx, 0: 1] = torch.LongTensor([char2Idx['<PAD>']])
            yield [wordSeqTensor, tagSeqTensor, wordSeqLengths, charSeqTensor, charSeqLengths, seq2NodeTensor, node2SeqTensor, adjMatrixTensor, gazNode2Idxs, gazNodeLengths, nodeNums, gazBlankState, fwbigramTensor, bwbigramTensor]
        if (totalSize // batchSize) * batchSize < totalSize:
            startId = (totalSize // batchSize) * batchSize
            lastBatchSize = totalSize - startId
            batchUtts = corpus.utterances[startId: totalSize]
            wordSeqLengths = torch.LongTensor(list(map(lambda utt: len(utt.tokens), batchUtts)))
            nodeNums = torch.LongTensor(lastBatchSize)
            maxSeqLength = wordSeqLengths.max()
            if self.useChar:
                charSeqLengths = torch.LongTensor(
                    [list(map(lambda tok: len(tok.chars), utt.tokens)) + [1] * (int(maxSeqLength) - len(utt.tokens)) for utt
                     in batchUtts])
                maxCharLength = charSeqLengths.max()
                charSeqTensor = autograd.Variable(torch.zeros((lastBatchSize, maxSeqLength, maxCharLength))).long()
            else:
                charSeqLengths = None
                maxCharLength = None
                charSeqTensor = None
            wordSeqTensor = autograd.Variable(torch.zeros((lastBatchSize, maxSeqLength))).long()
            if self.useBigram:
                fwbigramTensor = autograd.Variable(torch.zeros([lastBatchSize, maxSeqLength])).long()
                bwbigramTensor = autograd.Variable(torch.zeros([lastBatchSize, maxSeqLength])).long()
            else:
                fwbigramTensor = None
                bwbigramTensor = None
            tagSeqTensor = autograd.Variable(torch.zeros((lastBatchSize, maxSeqLength))).long()
            seq2NodeTensor = autograd.Variable(torch.zeros([lastBatchSize, maxSeqLength, 1])).long()
            node2SeqTensor = autograd.Variable(torch.zeros([lastBatchSize, maxSeqLength, 1])).long()
            gazNodeLengths = []
            gazNode2Idxs = []

            maxTotalNode = max([utt.totalNode for utt in batchUtts])
            maxMainNode = max([utt.mainNode for utt in batchUtts])
            for gazIdx in range(self.gaNum):
                batchMaxNodeLength = max([utt.gazGraph[gazIdx][0] for utt in batchUtts])
                gazNode2Idxs.append(autograd.Variable(torch.zeros([lastBatchSize, batchMaxNodeLength])).long())
                gazNodeLengths.append(autograd.variable(torch.zeros([lastBatchSize])).long())
            if self.gaNum > 0:
                gazBlankState = torch.zeros([lastBatchSize, maxTotalNode - maxMainNode])
            else:
                gazBlankState = None

            mainEdges = []
            for idx in range(lastBatchSize):
                nNode, node2seq, seq2node, edges = batchUtts[idx].mainGraph
                nodeNums[idx] = nNode
                node2SeqTensor[idx, :nNode, 0] = torch.LongTensor(node2seq)
                node2SeqTensor[idx, nNode:, 0] = maxSeqLength - 1
                seq2NodeTensor[idx, :wordSeqLengths[idx], 0] = torch.LongTensor(seq2node)
                mainEdges.append(edges)
            adjMatrixTensor = autograd.Variable(torch.zeros([lastBatchSize, maxTotalNode, maxTotalNode * self.edgeTypes]))

            for idx in range(lastBatchSize):
                mainTypes = len(mainEdges[idx])
                for typeIdx in range(len(mainEdges[idx])):
                    for edge in mainEdges[idx][typeIdx]:
                        adjMatrixTensor[idx, edge[0], maxTotalNode * typeIdx + edge[1]] = edge[2]

                for gazIdx in range(self.gaNum):
                    nNode, node2idx, edges = batchUtts[idx].gazGraph[gazIdx]
                    gazNodeLengths[gazIdx][idx] = nNode
                    gazNode2Idxs[gazIdx][idx][:nNode] = torch.LongTensor(node2idx)
                    for typeIdx in range(len(edges)):
                        for edge in edges[typeIdx]:
                            adjMatrixTensor[idx, edge[0], maxTotalNode * (mainTypes + typeIdx) + edge[1]] = edge[2]

                wordSeqTensor[idx, :wordSeqLengths[idx]] = torch.LongTensor(
                    [word2Idx.get(word.text, corpusMeta.unk) for word in batchUtts[idx].tokens])
                tagSeqTensor[idx, :wordSeqLengths[idx]] = torch.LongTensor(
                    [tag2Idx[word.tag] for word in batchUtts[idx].tokens])
                if self.useBigram:
                    fwbigramTensor[idx, :wordSeqLengths[idx]] = torch.LongTensor([fwbigram2idx.get(word.fwbigram, corpusMeta.unk) for word in batchUtts[idx].tokens])
                    bwbigramTensor[idx, :wordSeqLengths[idx]] = torch.LongTensor([bwbigram2idx.get(word.bwbigram, corpusMeta.unk) for word in batchUtts[idx].tokens])
                if self.useChar:
                    for wordIdx in range(wordSeqLengths[idx]):
                        charSeqTensor[idx, wordIdx, :charSeqLengths[idx, wordIdx]] = torch.LongTensor(
                            [char2Idx.get(char, corpusMeta.unk) for char in batchUtts[idx].tokens[wordIdx].chars])
                    for wordIdx in range(wordSeqLengths[idx], maxSeqLength):
                        charSeqTensor[idx, wordIdx, 0: 1] = torch.LongTensor([char2Idx['<PAD>']])
            yield [wordSeqTensor, tagSeqTensor, wordSeqLengths, charSeqTensor, charSeqLengths, seq2NodeTensor, node2SeqTensor, adjMatrixTensor, gazNode2Idxs, gazNodeLengths, nodeNums, gazBlankState, fwbigramTensor, bwbigramTensor]

    def getRawSentenceBatches(self, corpus, corpusMeta, batchSize):
        rawSentenceBatches = []
        totalSize = len(corpus.utterances)
        for batchId in range(totalSize // batchSize):
            batchUtts = corpus.utterances[batchId * batchSize: (batchId + 1) * batchSize]
            sentences = []
            for idx in range(batchSize):
                sentences.append([word.rawWord for word in batchUtts[idx].tokens])
            rawSentenceBatches.append(sentences)
        if len(rawSentenceBatches) * batchSize < totalSize:
            startId = len(rawSentenceBatches) * batchSize
            lastBatchSize = totalSize - startId
            batchUtts = corpus.utterances[startId: totalSize]
            sentences = []
            for idx in range(lastBatchSize):
                sentences.append([word.rawWord for word in batchUtts[idx].tokens])
            rawSentenceBatches.append(sentences)
        return rawSentenceBatches

    def negLogLikelihoodLoss(self, batchInput):
        wordSeqTensor, tagSeqTensor, wordSeqLengths, charSeqTensor, charSeqLengths, seq2NodeTensor, node2SeqTensor, adjMatrixTensor, gazNode2Idxs, gazNodeLengths, nodeNums, gazBlankState, fwbigramTensor, bwbigramTensor = batchInput
        batchSize = wordSeqTensor.shape[0]
        sentLength = wordSeqTensor.shape[1]
        maskTemp = torch.arange(1, sentLength + 1, dtype=torch.int64).view(1, sentLength).expand(batchSize, sentLength)
        if self.useGpu:
            maskTemp = move2cuda(maskTemp)
        mask = torch.le(maskTemp, wordSeqLengths.view(batchSize, 1).expand(batchSize, sentLength))
        if self.useGpu:
            mask = move2cuda(mask)
        if self.useChar:
            wordSeqEmbedding = self.dropout(self.wordEmbedding(wordSeqTensor, charSeqTensor, charSeqLengths))
        else:
            if self.useBigram:
                wordSeqEmbedding = self.dropout(torch.cat([self.wordEmbedding(wordSeqTensor), self.fwbigramEmbedding(fwbigramTensor), self.bwbigramEmbedding(bwbigramTensor)], 2))
            else:
                wordSeqEmbedding = self.dropout(self.wordEmbedding(wordSeqTensor))

        wordStateEmbedding = self.embStateLinear(wordSeqEmbedding)
        maxMainNodeLength = node2SeqTensor.shape[1]
        mainNodeState = torch.gather(wordStateEmbedding, 1, node2SeqTensor.expand(batchSize, maxMainNodeLength, wordStateEmbedding.shape[2]))
        if self.gaNum > 0:
            initNodeStateEmbedding = torch.cat([mainNodeState, gazBlankState.view(batchSize, -1, 1).expand(batchSize, -1, self.stateDim)], dim=1)
        else:
            initNodeStateEmbedding = mainNodeState
        startNodeIdx = nodeNums.clone()
        for gazIdx in range(self.gaNum):
            gazState = self.gaLinear[gazIdx](self.gaEmb[gazIdx](gazNode2Idxs[gazIdx]))
            gazMaskRaw = torch.arange(0, gazState.shape[1], dtype=torch.int64).view(1, gazState.shape[1], 1).expand(batchSize, gazState.shape[1], self.stateDim)
            if self.useGpu:
                gazMaskRaw = move2cuda(gazMaskRaw)
            gazMask = torch.where(gazMaskRaw < gazNodeLengths[gazIdx].view(batchSize, 1, 1), gazMaskRaw, gazNodeLengths[gazIdx].view(batchSize, 1, 1))
            if self.useGpu:
                gazMask = move2cuda(gazMask)
            gazMask = gazMask + startNodeIdx.view(batchSize, 1, 1).expand(batchSize, gazState.shape[1], self.stateDim)
            initNodeStateEmbedding.scatter_(1, gazMask, gazState)
            startNodeIdx = startNodeIdx + gazNodeLengths[gazIdx]

        nodeGraphEmbeddings = [initNodeStateEmbedding]
        for i in range(self.nLayer):
            nodeGraphEmbeddings.append(self.graphEmb[i](nodeGraphEmbeddings[i], adjMatrixTensor, adjMatrixTensor.shape[1]))
        nodeGraphEmbedding = nodeGraphEmbeddings[self.nLayer]
        wordGraphEmbedding = torch.gather(nodeGraphEmbedding, 1, seq2NodeTensor.expand([batchSize, sentLength, nodeGraphEmbedding.shape[2]]))

        if self.useRnn:
            rnnEmbedding = self.encoder(wordGraphEmbedding, wordSeqLengths)
            wordFeatures = self.logsoftmax(self.embFeatureLinear(rnnEmbedding))
        else:
            wordFeatures = self.logsoftmax(self.embFeatureLinear(wordGraphEmbedding))
        totalScore, scores = self.crf(wordFeatures, wordSeqLengths, mask)
        goldScore = self.crf.scoreSentence(tagSeqTensor, wordSeqLengths, scores, mask)

        return totalScore - goldScore

    def forward(self, batchInput, negMode=False):
        if negMode:
            return self.negLogLikelihoodLoss(batchInput)
        else:
            wordSeqTensor, tagSeqTensor, wordSeqLengths, charSeqTensor, charSeqLengths, seq2NodeTensor, node2SeqTensor, adjMatrixTensor0, gazNode2Idxs, gazNodeLengths, nodeNums, gazBlankState, fwbigramTensor, bwbigramTensor = batchInput
            batchSize = wordSeqTensor.shape[0]
            sentLength = wordSeqTensor.shape[1]
            if self.useChar:
                wordSeqEmbedding = self.dropout(self.wordEmbedding(wordSeqTensor, charSeqTensor, charSeqLengths))
            else:
                if self.useBigram:
                    wordSeqEmbedding = self.dropout(torch.cat(
                        [self.wordEmbedding(wordSeqTensor), self.fwbigramEmbedding(fwbigramTensor),
                         self.bwbigramEmbedding(bwbigramTensor)], 2))
                else:
                    wordSeqEmbedding = self.dropout(self.wordEmbedding(wordSeqTensor))
            wordStateEmbedding = self.embStateLinear(wordSeqEmbedding)
            maxNodeLength = node2SeqTensor.shape[1]
            mainNodeState = torch.gather(wordStateEmbedding, 1,
                                         node2SeqTensor.expand(batchSize, maxNodeLength, wordStateEmbedding.shape[2]))
            if self.gaNum > 0:
                initNodeStateEmbedding = torch.cat(
                    [mainNodeState, gazBlankState.view(batchSize, -1, 1).expand(batchSize, -1, self.stateDim)], dim=1)
            else:
                initNodeStateEmbedding = mainNodeState
            startNodeIdx = nodeNums.clone()
            for gazIdx in range(self.gaNum):
                gazState = self.gaLinear[gazIdx](self.gaEmb[gazIdx](gazNode2Idxs[gazIdx]))
                gazMaskRaw = torch.arange(0, gazState.shape[1], dtype=torch.int64).view(1, gazState.shape[1], 1).expand(
                    batchSize, gazState.shape[1], self.stateDim)
                if self.useGpu:
                    gazMaskRaw = move2cuda(gazMaskRaw)
                gazMask = torch.where(gazMaskRaw < gazNodeLengths[gazIdx].view(batchSize, 1, 1), gazMaskRaw,
                                      gazNodeLengths[gazIdx].view(batchSize, 1, 1))
                if self.useGpu:
                    gazMask = move2cuda(gazMask)
                gazMask = gazMask + startNodeIdx.view(batchSize, 1, 1).expand(batchSize, gazState.shape[1], self.stateDim)
                initNodeStateEmbedding.scatter_(1, gazMask, gazState)
                startNodeIdx = startNodeIdx + gazNodeLengths[gazIdx]

            adjMatrixTensor = adjMatrixTensor0

            nodeGraphEmbeddings = [initNodeStateEmbedding]
            for i in range(self.nLayer):
                nodeGraphEmbeddings.append(
                    self.graphEmb[i](nodeGraphEmbeddings[i], adjMatrixTensor, adjMatrixTensor.shape[1]))
            nodeGraphEmbedding = nodeGraphEmbeddings[self.nLayer]
            wordGraphEmbedding = torch.gather(nodeGraphEmbedding, 1,
                                              seq2NodeTensor.expand([batchSize, sentLength, nodeGraphEmbedding.shape[2]]))
            if self.useRnn:
                rnnEmbedding = self.encoder(wordGraphEmbedding, wordSeqLengths)
                wordFeatures = self.logsoftmax(self.embFeatureLinear(rnnEmbedding))
            else:
                wordFeatures = self.logsoftmax(self.embFeatureLinear(wordGraphEmbedding))
            bestScores, decodeIdx = self.crf.viterbiDecode(wordFeatures, wordSeqLengths)
            return bestScores, decodeIdx

