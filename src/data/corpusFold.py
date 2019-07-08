from collections import Counter
from tqdm import tqdm
from src.data.token import Token
from src.data.utterance import Utterance
import os
import re

class CorpusFold:
    def __init__(self, utterances, words, tags, chars):
        self.utterances = utterances
        self.words = words
        self.tags = tags
        self.chars = chars
        self.datasize = len(self.utterances)