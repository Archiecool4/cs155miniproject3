import nltk
import numpy as np
import os
import numpy as np
from IPython.display import HTML
from HMM import unsupervised_HMM
from HMM_helper import (
    parse_observations,
    sample_sentence,
)

from nltk.tokenize import TweetTokenizer
from collections import Counter, defaultdict
import re

import pickle

tokenizer = TweetTokenizer(preserve_case=False)

punctuation = ['.', ',', ';', ':', '!', '?', '\'']
filt = lambda s: s not in punctuation

outliers = ['\'gainst',
    '\'greeing',
    '\'scaped',
    '\'tis',
    '\'twixt'
]

words = []
text = []

syllables = defaultdict(list)

with open('data/Syllable_dictionary.txt') as f:
    for line in f:
        l = line.split()
        syllables[l[0]] += l[1:]

with open('data/shakespeare.txt') as f:
    for line in f:
        try:
            int(line)
        except:
            if line != '\n':
                tokens = tokenizer.tokenize(line)
                tokens = list(filter(filt, tokens))
                text.append(tokens)
                words += tokens

count = Counter(words)
unique = list(count.keys())

new_unique = []
ids_map = defaultdict(int)
for u in unique:
    word = '\'' + u
    if word in outliers:
        new_unique.append(word)
        ids_map[word] = len(new_unique) - 1
    else:
        new_unique.append(u)
        ids_map[u] = len(new_unique) - 1

ids = []
for i in text:
    line = []
    for j in i:
        word = '\'' + j
        if word in outliers:
            line.append(new_unique.index(word))
        else:
            line.append(new_unique.index(j))
    ids.append(line)

with open('obs.p', 'wb') as f:
    pickle.dump(ids, f)
    pickle.dump(ids_map, f)

hmm8 = unsupervised_HMM(ids, 10, 100)
pickle.dump(hmm8, open('hmm8.p', 'wb'))
print('Sample Sentence:\n====================')
print(sample_sentence(hmm8, ids_map, n_words=25))

hmm1 = unsupervised_HMM(ids, 1, 100)
pickle.dump(hmm1, open('hmm1.p', 'wb'))
print('\nSample Sentence:\n====================')
print(sample_sentence(hmm1, ids_map, n_words=25))

hmm2 = unsupervised_HMM(ids, 2, 100)
pickle.dump(hmm2, open('hmm2.p', 'wb'))
print('\nSample Sentence:\n====================')
print(sample_sentence(hmm2, ids_map, n_words=25))

hmm4 = unsupervised_HMM(ids, 4, 100)
pickle.dump(hmm4, open('hmm4.p', 'wb'))
print('\nSample Sentence:\n====================')
print(sample_sentence(hmm4, ids_map, n_words=25))

hmm16 = unsupervised_HMM(ids, 16, 100)
pickle.dump(hmm16, open('hmm16.p', 'wb'))
print('\nSample Sentence:\n====================')
print(sample_sentence(hmm16, ids_map, n_words=25))
