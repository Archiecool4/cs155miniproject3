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

hmm8 = pickle.load(open('hmm8.p', 'rb'))
hmm1 = pickle.load(open('hmm1.p', 'rb'))
hmm2 = pickle.load(open('hmm2.p', 'rb'))
hmm4 = pickle.load(open('hmm4.p', 'rb'))
hmm16 = pickle.load(open('hmm16.p', 'rb'))

ids = None
ids_map = None

with open('obs.p', 'rb') as f:
    ids = pickle.load(f)
    ids_map = pickle.load(f)

# print('Sample Sentence:\n====================')
# print(sample_sentence(hmm8, ids_map, n_words=5))
# print('\nSample Sentence:\n====================')
# print(sample_sentence(hmm1, ids_map, n_words=5))
# print('\nSample Sentence:\n====================')
# print(sample_sentence(hmm2, ids_map, n_words=5))
# print('\nSample Sentence:\n====================')
# print(sample_sentence(hmm4, ids_map, n_words=5))
print('\nSample Sentence:\n====================')
# print(sample_sentence(hmm16, ids_map, n_words=5))

for i in range(14):
    print(sample_sentence(hmm16, ids_map, n_words=5))

