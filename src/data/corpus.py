from collections import Counter
from tqdm import tqdm
from src.data.token import Token
from src.data.utterance import Utterance
import os
import re
from src.utils.globalVariable import GLOBAL_VARIABLE

class Corpus:
    def __init__(self, path, useNormalizedWord, removeEntities=[]):
        self.utterances, self.words, self.tags, self.chars, self.fwbigrams, self.bwbigrams = self.loadTrainData(path, useNormalizedWord, removeEntities=removeEntities)
        self.datasize = len(self.utterances)
        self.prepared = False

    def loadTrainData(self, path, useNormalizedWord, removeEntities):
        path += '.bieos'
        words = Counter()
        tags = Counter()
        chars = Counter()
        fwbigrams = Counter()
        bwbigrams = Counter()
        utterances = []
        length = 0

        print("Start loading data from {}...".format(path))
        with open(path, 'r', encoding='utf-8') as f:
            tokens = []
            for line in tqdm(f.readlines(), disable=GLOBAL_VARIABLE.DISABLE_TQDM):
                line = line.strip('\n')
                if line == "":
                    if len(tokens) > 0:
                        for tokenId in range(len(tokens)):
                            if tokenId + 1 < len(tokens):
                                tokens[tokenId].fwbigram = tokens[tokenId].text + tokens[tokenId + 1].text
                            else:
                                tokens[tokenId].fwbigram = '<END>'
                            if tokenId > 0:
                                tokens[tokenId].bwbigram = tokens[tokenId].text + tokens[tokenId - 1].text
                            else:
                                tokens[tokenId].bwbigram = '<START>'
                            fwbigrams.update([tokens[tokenId].fwbigram])
                            bwbigrams.update([tokens[tokenId].bwbigram])
                        utterances.append(Utterance(tokens))
                        length += len(tokens)
                    tokens = []
                    continue
                try:
                    word, tag = line.split('\t')
                except:
                    word = line
                    tag = None
                if useNormalizedWord:
                    normalizedWord = re.sub('[1-9]', '0', word)
                else:
                    normalizedWord = word
                tags.update([tag])
                tokens.append(Token(normalizedWord, line, word, tag))
                words.update([normalizedWord])
                chars.update([c for c in normalizedWord])
            if len(tokens) > 0:
                utterances.append(Utterance(tokens))
        print("{}: {} utterances, {} words, {} tags, {} chars, {:.1f} avg tokens".format(path, len(utterances), len(words), len(tags), len(chars), length / len(utterances)))
        return utterances, words, tags, chars, fwbigrams, bwbigrams

    def update(self, newCorpus):
        self.utterances += newCorpus.utterances
        self.words += newCorpus.words
        self.tags += newCorpus.tags
        self.chars += newCorpus.chars
        self.datasize += newCorpus.datasize


