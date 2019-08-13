def buildMainTrival(tokens):
    seqLen = len(tokens)
    node2seq = [i for i in range(seqLen)]
    seq2node = [i for i in range(seqLen)]
    edges = [[], []]
    for i in range(seqLen):
        if i > 0:
            edges[0].append([seq2node[i], seq2node[i-1], 1])
        if i < seqLen - 1:
            edges[1].append([seq2node[i], seq2node[i+1], 1])
    return seqLen, node2seq, seq2node, edges

def buildDictWithAbstractNode(startNode, seq2node, matches):
    node2idx = [2, 3]
    edges = [[], []]
    countMode = False
    if len(seq2node) == 0:
        countMode = True
    for match in matches:
        if not countMode:
            edges[0].append([startNode, seq2node[match[0]], 1])
            edges[1].append([seq2node[match[0]], startNode, 1])
            edges[0].append([seq2node[match[1]], startNode + 1, 1])
            edges[1].append([startNode + 1, seq2node[match[1]], 1])
        for i in range(match[0], match[1]):
            if i < match[1] - 1 and not countMode:
                edges[0].append([seq2node[i], seq2node[i + 1], 1])
            if i > match[0] and not countMode:
                edges[1].append([seq2node[i], seq2node[i - 1], 1])
    return 2, node2idx, edges
