from .trie import Trie
from tqdm import tqdm
import re
import numpy as np
from src.utils.globalVariable import GLOBAL_VARIABLE

class Gazetteer:
    def __init__(self, name, path, useNormalizedWord,  embDim, method, space, matchIgnoreCase, embedding=None, lineSep=' ', ratio=1):
        self.name = name
        self.useNormalizedWord = useNormalizedWord
        self.trie = Trie()
        self.ratio = ratio
        self.space = space
        self.matchIgnoreCase = matchIgnoreCase
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx2Word = ['<PAD>', '<UNK>', '<START>', '<END>']
        self.readGazetter(path, lineSep=lineSep)
        self.embDim = embDim
        self.wordEmbedding = np.empty([len(self.word2idx), self.embDim])
        self.word2emb = {}
        if embedding is not None:
            self.readEmbedding(embedding, lineSep=lineSep)
        self.getEmbMatrix()
        self.method = method

    def getEmbMatrix(self):
        scale = np.sqrt(3.0 / self.embDim)
        if len(self.word2emb) != 0:
            for wordIdx in range(len(self.word2idx)):
                if self.idx2Word[wordIdx] in self.word2emb:
                    self.wordEmbedding[wordIdx, :] = self.word2emb[self.idx2Word[wordIdx]]
                else:
                    self.wordEmbedding[wordIdx, :] = np.random.uniform(-scale, scale, [1, self.embDim])
        else:
            for wordIdx in range(len(self.word2idx)):
                self.wordEmbedding[wordIdx, :] = np.random.uniform(-scale, scale, [1, self.embDim])
        self.wordEmbedding[0, :] = np.zeros([1, self.embDim])

    def readEmbedding(self, path, lineSep=' '):
        print('Reading gazetter {} embeddings...'.format(self.name))
        with open(path, 'r', encoding='utf-8') as fh:
            for line in tqdm(fh.readlines(), disable=GLOBAL_VARIABLE.DISABLE_TQDM):
                line = line.strip('\n')
                if line == '':
                    continue
                if line[-1] == lineSep:
                    line = line[:-1]
                items = line.split(lineSep)
                word = items[0]
                if self.matchIgnoreCase:
                    word = word.lower()
                if self.useNormalizedWord:
                    word = re.sub('[1-9]', '0', word)
                if word in self.word2idx:
                    embedding = np.empty([1, self.embDim])
                    embedding[:] = items[1:]
                    self.word2emb[word] = embedding
                    assert self.embDim == len(items) - 1

    def readGazetter(self, path, lineSep=' ', minLength = 0):
        counter = 0
        total = len(open(path, 'r', encoding='utf-8').readlines())
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip('\n')
                if line == '':
                    continue
                counter += 1
                if counter > total * self.ratio:
                    break
                word = line.split(lineSep)[0]
                if self.useNormalizedWord:
                    word = re.sub('[1-9]', '0', word)
                if self.matchIgnoreCase:
                    word = word.lower()
                if self.space != '':
                    charList = word.split(self.space)
                else:
                    charList = [c for c in word]
                if len(charList) >= minLength:
                    if not word in self.word2idx:
                        self.trie.insert(charList)
                        self.word2idx[word] = len(self.word2idx)
                        self.idx2Word.append(word)

    def enumerateMatchList(self, word_list):
        result = []
        for startPos in range(len(word_list)):
            match_list = self.trie.search(word_list[startPos: ], ignoreCase=self.matchIgnoreCase)
            if match_list is None:
                continue
            for match in match_list:
                result.append([startPos, startPos + len(match), self.word2idx.get(self.space.join(match))])
        return result

    def matchCorpus(self, corpus):
        for utt in corpus.utterances:
            utt.gazMatch[self.name] = self.enumerateMatchList([tok.text for tok in utt.tokens])





