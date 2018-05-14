"""
Constructing and loading dictionaries
"""
import numpy
from collections import OrderedDict

def build_dictionary(text):
    """
    Build a dictionary
    text: list of sentences (pre-tokenized)
    """
    wordcount = OrderedDict()
    for cc in text:
        words = cc.split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 0
            wordcount[w] += 1
    words = wordcount.keys()
    freqs = wordcount.values()
    sorted_idx = numpy.argsort(list(freqs))[::-1]

    worddict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        #next(numpy.itertools.islice(line_1.values(), 1, 2))
        worddict[list(words)[sidx]] = idx + 2   # 0: <eos>, 1: <unk>

    return worddict, wordcount