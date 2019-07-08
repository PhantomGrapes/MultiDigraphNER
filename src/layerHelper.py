import torch
import math
from torch import nn
from src.models.crfs.linearChainCRF import LinearChainCRF
from src.models.encoders.simpleEmbedding import SimpleEmbedding
from src.models.encoders.charWordEmbedding import CharWordEmbedding
from src.models.encoders.GGNN import GGNN
from src.models.encoders.simpleLstm import SimpleLSTM
from src.models.encoders.propogator import GRUProp
from src.utils.gpu_tools import move2cuda

class LayerUtils:
    def __init__(self, Config, corpusMeta):
        self.Config = Config
        self.tagSize = len(corpusMeta.tag2Idx)
        self.wordSize = len(corpusMeta.word2idx)
        self.wordEmbedding = corpusMeta.wordEmbedding
        self.fwbigramEmbedding = corpusMeta.fwbigramEmbedding
        self.bwbigramEmbedding = corpusMeta.bwbigramEmbedding
        self.charSize = len(corpusMeta.char2Idx)
        self.charEmbeddingDim = corpusMeta.charEmbeddingDim
        self.wordEmbeddingDim = self.getWordEmbeddingDim()
        self.maxSentLength = corpusMeta.maxSentLength
        gazetters = corpusMeta.gazetters
        self.gaNum = len(gazetters)
        if self.gaNum > 0:
            self.gaEmbs = []
            self.gaEmbDims = []
            for i in range(self.gaNum):
                self.gaEmbs.append(gazetters[i].wordEmbedding)
                self.gaEmbDims.append(gazetters[i].embDim)
        self.initGa = [1 for _ in range(self.gaNum * 2 + 2)]

    def getGaEmbs(self):
        embs = []
        for i in range(self.gaNum):
            embs.append(nn.Embedding.from_pretrained(torch.FloatTensor(self.gaEmbs[i]), freeze=False))
        return embs

    def getGaEmb2States(self):
        linears = []
        for i in range(self.gaNum):
            linears.append(nn.Linear(self.gaEmbDims[i], self.getGraphDim()))
        return linears

    def getGaRelMatrixes(self):
        matrix = nn.Parameter(torch.Tensor(self.getGraphDim(), self.getGraphDim()))
        stdv = 1. / math.sqrt(matrix.size(1))
        matrix.data.uniform_(-stdv, stdv)
        return nn.Parameter(matrix.view(1, self.getGraphDim(), self.getGraphDim()))

    def getEdgeTypes(self):
        n = 0
        if self.Config.model.graph_emb.build_main in ['trival']:
            n += 2
        if self.Config.model.get("graph_emb", None) is not None and self.Config.model.graph_emb.get('gazetter', None) is not None:
            for name in self.Config.model.graph_emb.gazetter.to_dict():
                if name != 'get_tag':
                    gaItem = self.Config.model.graph_emb.gazetter.get(name)
                    if gaItem['method'].lower() in ['abnode']:
                        n += 2
        if self.Config.model.graph_emb.attention:
            n += 1
        return n

    def getGraphDim(self):
        return self.Config.model.graph_emb.state_dim

    def getEmb2Feature(self):
        if self.Config.model.graph_emb.use_rnn:
            return nn.Linear(self.Config.model.rnn_hidden_size, self.tagSize)
        else:
            return nn.Linear(self.getGraphDim(), self.tagSize)

    def getEmbStateLinear(self):
        return nn.Linear(self.getWordEmbeddingDim(), self.getGraphDim())

    def getWordEmbeddingDim(self):
        if self.Config.model.use_char:
            return self.Config.model.word_embedding_dim + self.Config.model.char_dim
        else:
            if self.Config.model.use_bigram:
                return self.Config.model.word_embedding_dim + 2 * self.Config.model.bigram_dim
            else:
                return self.Config.model.word_embedding_dim

    def getInitHidden(self, rnnType, numLayers, useBiRNN, batchSize, hiddenDim):
        if self.Config.train.get('rnn_hidden_initalization', 'none') == "none":
            hidden = None
        elif self.Config.train.get('rnn_hidden_initalization', 'none') == "randn":
            if useBiRNN:
                hiddenDim = hiddenDim // 2
            if rnnType == "lstm":
                hidden = (torch.randn(numLayers * 2 if useBiRNN else numLayers, batchSize, hiddenDim),
                        torch.randn(numLayers * 2 if useBiRNN else numLayers, batchSize, hiddenDim))
                if self.Config.use_gpu:
                    hidden = move2cuda(hidden)
            elif rnnType == "gru":
                hidden = torch.randn(numLayers * 2 if useBiRNN else numLayers, batchSize, hiddenDim)
                if self.Config.use_gpu:
                    hidden = move2cuda(hidden)
            else:
                print("Invalid rnn type {}".format(rnnType))
                exit(1)
            if self.Config.use_gpu:
                hidden = move2cuda(hidden)
        elif self.Config.train.get('rnn_hidden_initalization', 'none') == "normal":
            if useBiRNN:
                hiddenDim = hiddenDim // 2
            if rnnType == "lstm":
                hidden = (torch.normal(mean=torch.zeros(numLayers * 2 if useBiRNN else numLayers, batchSize, hiddenDim)),
                          torch.normal(mean=torch.zeros(numLayers * 2 if useBiRNN else numLayers, batchSize, hiddenDim)))
                if self.Config.use_gpu:
                    hidden = move2cuda(hidden)
            elif rnnType == "gru":
                hidden = torch.normal(mean=torch.zeros(numLayers * 2 if useBiRNN else numLayers, batchSize, hiddenDim))
                if self.Config.use_gpu:
                    hidden = move2cuda(hidden)
            else:
                print("Invalid rnn type {}".format(rnnType))
                exit(1)

        return hidden

    def getRNN(self, rnnType, useBiRNN, inputDim, hiddenDim, numLayers, dropout):
        if useBiRNN:
            hiddenDim = hiddenDim // 2
        if rnnType == "lstm":
            return nn.LSTM(inputDim, hiddenDim, numLayers, bidirectional=useBiRNN)
        elif rnnType == "gru":
            return nn.GRU(inputDim, hiddenDim, numLayers, bidirectional=useBiRNN)
        else:
            print("Invalide rnn type {}".format(rnnType))
            exit(1)

    def getEncoderRNN(self):
        if self.Config.model.model_type in ["graph"]:
            return (self.getRNN(self.Config.model.rnn_type, self.Config.model.use_birnn,
                                self.getGraphDim(), self.Config.model.rnn_hidden_size,
                               self.Config.model.rnn_num_layers, self.Config.model.drop_out),
                    lambda currentBatchSize: self.getInitHidden(self.Config.model.rnn_type, self.Config.model.rnn_num_layers,
                                                                self.Config.model.use_birnn, currentBatchSize,
                                                                self.Config.model.rnn_hidden_size))

    def getEncoderLinear(self):
        if self.Config.model.model_type in ["graph"]:
            return nn.Linear(self.Config.model.rnn_hidden_size, self.tagSize)

    def getActivation(self):
        if self.Config.model.get('activation', 'none') == "none":
            return lambda x: x
        elif self.Config.model.get('activation', 'none') == "relu":
            return torch.nn.ReLU()
        elif self.Config.model.get('activation', 'none') == "tanh":
            return torch.nn.Tanh()
        elif self.Config.model.get('activation', 'none') == "sigmoid":
            return torch.nn.Sigmoid()
        else:
            print("Invalide activation type {}".format(self.Config.model.get('activation', 'none') ))
            exit(1)

    def getDropOut(self):
        return nn.Dropout(self.Config.model.drop_out)

    def getTransitionMatrix(self):
        if self.Config.train.get('transition_matrix_initalization', 'randn') == 'randn':
            transitions = nn.Parameter(torch.randn(self.tagSize, self.tagSize))
        elif self.Config.train.get('transition_matrix_initalization', 'randn') == 'normal':
            transitions = nn.Parameter(torch.normal(mean=torch.zeros(self.tagSize, self.tagSize)))
        transitions.data[self.Config.data.TAG_START_ID, :] = -10000
        transitions.data[:, self.Config.data.TAG_EOS_ID] = -10000
        transitions.data[:, 0] = -10000
        transitions.data[0, :] = -10000
        return transitions

    def getInitAlpha(self, batchSize):
        initAlpha = torch.full((batchSize, self.tagSize), -10000)
        for idx in range(batchSize):
            initAlpha[idx][self.Config.data.TAG_START_ID] = 0
        if self.Config.use_gpu:
            initAlpha = move2cuda(initAlpha)
        return initAlpha

    def getInitAlphaVector(self):
        initAlpha = torch.full((self.tagSize), -10000)
        initAlpha[self.Config.data.TAG_START_ID] = 0
        if self.Config.use_gpu:
            initAlpha = move2cuda(initAlpha)
        return initAlpha

    def getInitAlphaWithBatchSize(self):
        return lambda batchSize: self.getInitAlpha(batchSize)

    def getTagSize(self):
        return self.tagSize

    def getTagEndId(self):
        return self.Config.data.TAG_EOS_ID

    def getTagStartId(self):
        return self.Config.data.TAG_START_ID

    def getUseGpu(self):
        return self.Config.use_gpu

    def getEmbeddingParameter(self):
        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.wordEmbedding), freeze=False)
        return embedding

    def getBigramParameter(self):
        fwembedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.fwbigramEmbedding), freeze=False)
        bwembedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.bwbigramEmbedding), freeze=False)
        return fwembedding, bwembedding

    def getCharEmbeddingParameter(self):
        return nn.Embedding(self.charSize, self.charEmbeddingDim, padding_idx=0)

    def getCharConv(self):
        return nn.Conv1d(self.charEmbeddingDim, self.Config.model.char_dim, kernel_size=3, padding=1)

    def getCharRNN(self):
        return nn.LSTM(self.charEmbeddingDim, self.Config.model.char_dim, bidirectional=False)

    def getPropagator(self, dim, edgeType):
        if self.Config.model.graph_emb.prop_type.lower() == 'gru':
            return GRUProp(dim, edgeType, self.getUseGpu())


class LayerHelper:
    def __init__(self, Config, layerUtils):
        self.Config = Config
        self.layerUtils = layerUtils

    def getEncoder(self):
        if self.Config.model.model_type in ["graph"]:
            return SimpleLSTM(self.layerUtils)

    def getEmbFeatureLinear(self):
        return self.layerUtils.getEmb2Feature()

    def getGraphEmbedding(self):
        return GGNN(self.layerUtils)

    def getCRF(self):
        if self.Config.model.model_type in ["graph"]:
            return LinearChainCRF(self.layerUtils)

    def getLinearCRF(self):
        return LinearChainCRF(self.layerUtils)

    def getWordEmbedding(self):
        if self.Config.model.model_type in ["graph"]:
            if self.Config.model.use_char:
                return CharWordEmbedding(self.layerUtils)
            else:
                return SimpleEmbedding(self.layerUtils)

    def getBigramEmbedding(self):
        return self.layerUtils.getBigramParameter()
