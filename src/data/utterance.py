from src.utils.buildGraph import *

class Utterance:
    def __init__(self, tokens):
        self.tokens = tokens
        self.uttLen = len(tokens)
        self.entities = []
        self.keepTokens = [i for i in range(len(tokens))]
        self.gazMatch = {}
        self.seq2node = []
        self.mainGraph = None
        self.gazGraph = []
        self.totalNode = 0
        self.mainNode = 0

    def buildMainGraph(self, method):
        if method.lower() == 'trival':
            nNode, node2seq, seq2node, edges = buildMainTrival(self.tokens)
        self.seq2node = seq2node
        return nNode, node2seq, seq2node, edges

    def buildGazetterGraph(self, method, gaName, startNode):
        if method.lower() == 'abnode':
            return buildDictWithAbstractNode(startNode, self.seq2node, self.gazMatch[gaName])
