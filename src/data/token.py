class Token:
    def __init__(self, text, rawLine, rawWord, tag=None, elmo=None, pos=None, constraint=None):
        self.text = text
        self.tag = tag
        self.rawLine = rawLine
        self.rawWord = rawWord
        self.elmo = elmo
        self.pos = pos
        self.chars = [c for c in text]
        self.marginalProba = None
        self.constraint = constraint
        self.fwbigram = None
        self.bwbigram = None
