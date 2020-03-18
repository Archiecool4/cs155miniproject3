import nltk
import numpy as np
import os
import numpy as np
from IPython.display import HTML
from HMM import unsupervised_HMM
from HMM_helper import (
    text_to_wordcloud,
    states_to_wordclouds,
    parse_observations,
    sample_sentence,
    visualize_sparsities,
    animate_emission
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
hmm32 = pickle.load(open('hmm32.p', 'rb'))
hmm64 = pickle.load(open('hmm64.p', 'rb'))

ids = None
ids_map = None

with open('obs.p', 'rb') as f:
    ids = pickle.load(f)
    ids_map = pickle.load(f)

text = ''

for i in range(14):
    new = sample_sentence(hmm64, ids_map, n_words=5)
    print(new)
    text += new

obs_map = []

# for key, value in ids_map.items():
#     if key in text:
#         obs_map.append(key)

obs_map = dict(filter(lambda x: x[0] in text, ids_map.items()))

wordcloud = text_to_wordcloud(text, title='sonnet')

visualize_sparsities(hmm64, O_max_cols=50)

anim = animate_emission(hmm64, obs_map, M=64)
HTML(anim.to_html5_video())

