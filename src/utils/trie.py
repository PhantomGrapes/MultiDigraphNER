import collections
class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        current = self.root
        for letter in word:
            current = current.children[letter]
        current.is_word = True

    def search(self, word, ignoreCase=False):
        wordList = []
        current = self.root
        chars = []
        for letter in word:
            if ignoreCase:
                letter = letter.lower()
            chars.append(letter)
            current = current.children.get(letter)

            if current is None:
                return wordList
            if current.is_word:
                wordList.append(chars.copy())

    def startsWith(self, prefix):
        current = self.root
        for letter in prefix:
            current = current.children.get(letter)
            if current is None:
                return False
        return True


