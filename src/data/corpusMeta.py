from tqdm import tqdm
import random
import re
import numpy as np
from src.utils.globalVariable import GLOBAL_VARIABLE

def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    if root_sum_square > 0:
        return vec/root_sum_square
    else:
        return vec

class CorpusMeta:
    def __init__(self, corpus, Config):
        self.wordEmbedding = np.empty([len(corpus.words) + 2, Config.model.word_embedding_dim])
        self.fwbigramEmbedding = np.empty([len(corpus.fwbigrams) + 2, Config.model.bigram_dim])
        self.bwbigramEmbedding = np.empty([len(corpus.fwbigrams) + 2, Config.model.bigram_dim])
        self.useNormalizedWord = Config.data.use_normalized_word
        self.idx2Word = self.getIdx2Word(corpus.words, Config)
        self.word2idx = {self.idx2Word[idx]: idx for idx in range(len(self.idx2Word))}
        self.idx2Fwbigram, self.fwbigramEmbedding, self.idx2Bwbigram, self.bwbigramEmbedding = self.getIdx2Bigram(corpus.fwbigrams, corpus.bwbigrams, Config, self.fwbigramEmbedding, self.bwbigramEmbedding)
        self.fwbigram2idx = {self.idx2Fwbigram[idx]: idx for idx in range(len(self.idx2Fwbigram))}
        self.bwbigram2idx = {self.idx2Bwbigram[idx]: idx for idx in range(len(self.idx2Bwbigram))}
        self.idx2Tag = self.getIdx2Tag(corpus.tags, Config)
        self.tag2Idx = {self.idx2Tag[idx]: idx for idx in range(len(self.idx2Tag))}
        self.idx2Char = self.getIdx2Char(corpus.chars, Config)
        self.char2Idx = {self.idx2Char[idx]: idx for idx in range(len(self.idx2Char))}
        self.unk = Config.data.UNK_ID
        self.charEmbeddingDim = self.getCharEmbeddingDim()
        self.maxSentLength = max([len(utt.tokens) for utt in corpus.utterances])
        self.gazetters = []

    def updateMaxSentLength(self, corpus):
        self.maxSentLength = max([self.maxSentLength] +  [len(utt.tokens) for utt in corpus.utterances])

    def getCharEmbeddingDim(self):
        # return len(self.char2Idx) * 3
        return 30

    def getIdx2Word(self, words, Config, norm=False):
        validWords = list(words.keys())
        validWords.insert(Config.data.PAD_ID, '<PAD>')
        validWords.insert(Config.data.UNK_ID, '<UNK>')
        scale = np.sqrt(3.0 / Config.model.word_embedding_dim)
        # for wordItem in words.most_common():
        #     if wordItem[1] <= Config.data.word_threshold:
        #         break
        #     validWords.append(wordItem[0])

        if Config.data.get('word_embedding', False):
            pretrainDict = {}
            with open(Config.data.word_embedding, 'r', encoding='utf-8') as fh:
                print('Reading word embeddings...')
                for line in tqdm(fh.readlines(), disable=GLOBAL_VARIABLE.DISABLE_TQDM):
                    items = line.split()
                    assert len(items) == Config.model.word_embedding_dim + 1
                    word = items[0]
                    if self.useNormalizedWord:
                        word = re.sub('[1-9]', '0', word)
                    if not word in pretrainDict:
                        embedding = np.empty([1, Config.model.word_embedding_dim])
                        embedding[:] = items[1:]
                        pretrainDict[word] = embedding
            for wordIdx in range(len(validWords)):
                if validWords[wordIdx] in pretrainDict:
                    self.wordEmbedding[wordIdx, :] = norm2one(pretrainDict[validWords[wordIdx]]) if norm else pretrainDict[validWords[wordIdx]]
                elif validWords[wordIdx].lower() in pretrainDict:
                    self.wordEmbedding[wordIdx, :] = norm2one(pretrainDict[validWords[wordIdx].lower()]) if norm else pretrainDict[validWords[wordIdx].lower()]
                else:
                    self.wordEmbedding[wordIdx, :] = np.random.uniform(-scale, scale,
                                                                       [1, Config.model.word_embedding_dim])
        else:
            for wordIdx in range(len(validWords)):
                self.wordEmbedding[wordIdx, :] = np.random.uniform(-scale, scale, [1, Config.model.word_embedding_dim])
        self.wordEmbedding[0, :] = np.zeros([1, Config.model.word_embedding_dim])

        return validWords

    def getIdx2Bigram(self, fwwords, bwwords, Config, fwembedding, bwembedding, norm=False):
        fwvalidWords = list(fwwords.keys())
        fwvalidWords.insert(Config.data.PAD_ID, '<PAD>')
        fwvalidWords.insert(Config.data.UNK_ID, '<UNK>')

        bwvalidWords = list(bwwords.keys())
        bwvalidWords.insert(Config.data.PAD_ID, '<PAD>')
        bwvalidWords.insert(Config.data.UNK_ID, '<UNK>')
        scale = np.sqrt(3.0 / Config.model.bigram_dim)
        # for wordItem in words.most_common():
        #     if wordItem[1] <= Config.data.word_threshold:
        #         break
        #     validWords.append(wordItem[0])

        if Config.data.get('bigram_embedding', False):
            pretrainDict = {}
            with open(Config.data.bigram_embedding, 'r', encoding='utf-8') as fh:
                print('Reading bigram embeddings...')
                for line in tqdm(fh.readlines(), disable=GLOBAL_VARIABLE.DISABLE_TQDM):
                    items = line.split()
                    assert len(items) == Config.model.bigram_dim + 1
                    word = items[0]
                    if not word in pretrainDict:
                        embedding = np.empty([1, Config.model.bigram_dim])
                        embedding[:] = items[1:]
                        pretrainDict[word] = embedding
            for wordIdx in range(len(fwvalidWords)):
                if fwvalidWords[wordIdx] in pretrainDict:
                    fwembedding[wordIdx, :] = norm2one(pretrainDict[fwvalidWords[wordIdx]]) if norm else pretrainDict[fwvalidWords[wordIdx]]
                elif fwvalidWords[wordIdx].lower() in pretrainDict:
                    fwembedding[wordIdx, :] = norm2one(pretrainDict[fwvalidWords[wordIdx].lower()]) if norm else pretrainDict[fwvalidWords[wordIdx].lower()]
                else:
                    fwembedding[wordIdx, :] = np.random.uniform(-scale, scale,
                                                                       [1, Config.model.bigram_dim])

            for wordIdx in range(len(bwvalidWords)):
                if bwvalidWords[wordIdx] in pretrainDict:
                    bwembedding[wordIdx, :] = norm2one(pretrainDict[bwvalidWords[wordIdx]]) if norm else pretrainDict[bwvalidWords[wordIdx]]
                elif bwvalidWords[wordIdx].lower() in pretrainDict:
                    bwembedding[wordIdx, :] = norm2one(pretrainDict[bwvalidWords[wordIdx].lower()]) if norm else pretrainDict[bwvalidWords[wordIdx].lower()]
                else:
                    bwembedding[wordIdx, :] = np.random.uniform(-scale, scale,
                                                                       [1, Config.model.bigram_dim])
        else:
            for wordIdx in range(len(fwvalidWords)):
                fwembedding[wordIdx, :] = np.random.uniform(-scale, scale, [1, Config.model.bigram_dim])
            for wordIdx in range(len(bwvalidWords)):
                bwembedding[wordIdx, :] = np.random.uniform(-scale, scale, [1, Config.model.bigram_dim])
        fwembedding[0, :] = np.zeros([1, Config.model.bigram_dim])
        bwembedding[0, :] = np.zeros([1, Config.model.bigram_dim])

        return fwvalidWords, fwembedding, bwvalidWords, bwembedding

    def getIdx2Tag(self, tags, Config):
        validTags =  [tag[0] for tag in tags.items()
                if len(Config.data.keep_labels) == 0 or tag[0] in Config.data.keep_labels]
        validTags.insert(Config.data.PAD_ID, '<PAD>')
        validTags.append('<START>')
        validTags.append('<EOS>')
        return validTags

    def getIdx2Char(self, chars, Config):
        validChars =  [char[0] for char in chars.items()]
        validChars.insert(Config.data.PAD_ID, '<PAD>')
        validChars.insert(Config.data.UNK_ID, '<UNK>')
        return validChars
