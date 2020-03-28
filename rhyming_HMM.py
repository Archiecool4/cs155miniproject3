import nltk
import numpy as np
import os
import random
from IPython.display import HTML
from HMM import unsupervised_HMM
from HMM_helper import (
    text_to_wordcloud,
    states_to_wordclouds,
    parse_observations,
    sample_sentence,
    sample_rhyming_sentence,
    visualize_sparsities,
    animate_emission,
    obs_map_reverser
)

from nltk.tokenize import TweetTokenizer
from collections import Counter, defaultdict
import re
import pronouncing
import pickle


####################
#      SETUP       #
####################
hmm8 = pickle.load(open('hmm8.p', 'rb'))
hmm1 = pickle.load(open('hmm1.p', 'rb'))
hmm2 = pickle.load(open('hmm2.p', 'rb'))
hmm4 = pickle.load(open('hmm4.p', 'rb'))
hmm16 = pickle.load(open('hmm16.p', 'rb'))
hmm32 = pickle.load(open('hmm32.p', 'rb'))
hmm64 = pickle.load(open('hmm64.p', 'rb'))

# Load ids and id map
ids = None
ids_map = None
with open('obs.p', 'rb') as f:
    ids = pickle.load(f)
    ids_map = pickle.load(f)


####################
#    GET RHYMES    #
####################
# words = list(ids_map.keys())
# num_words = len(words)
#
# # Create dictionary of rhyming words
# rhymes_map = dict()
# for i in range(num_words):
#     rhymes_map[words[i]] = list()
#     for j in range(num_words):
#         if words[j] in pronouncing.rhymes(words[i]):
#             rhymes_map[words[i]].append(words[j])

# Save dictionary for fast access later
# with open('rhymes_map.pickle', 'wb') as handle:
#     pickle.dump(rhymes_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('rhymes_map.pickle', 'rb') as handle:
    rhymes_map = pickle.load(handle)

####################
#   GENERATE POEM  #
####################
text = ''
# Get reverse map.
obs_map_r = obs_map_reverser(ids_map)

# Generate first 3 stanzas
for i in range(3):
    # Get two seeds for each stanza
    while True:
        w1 = random.choice(list(ids_map.values()))
        if len(rhymes_map[obs_map_r[w1]]) != 0:
            break
    while True:
        w2 = random.choice(list(ids_map.values()))
        if len(rhymes_map[obs_map_r[w2]]) != 0:
            break
    seeds = [w1, w2]
    # Generate 4 lines for each stanza
    for j in range(4):
        if j == 0 or j == 1:
            text += sample_rhyming_sentence(hmm64, ids_map, seeds[j], n_words = 7)
            text += '\n'
        else:
            # Rhyming word
            rhyme = random.choice(rhymes_map[obs_map_r[seeds[j - 2]]])
            text += sample_rhyming_sentence(hmm64, ids_map, ids_map[rhyme], n_words = 7)
            text += '\n'

# Generate last 2 lines
while True:
    w3 = random.choice(list(ids_map.values()))
    if len(rhymes_map[obs_map_r[w3]]) != 0:
        break
text += sample_rhyming_sentence(hmm64, ids_map, w3, n_words = 7)
text += '\n'
rhyme = random.choice(rhymes_map[obs_map_r[w3]])
text += sample_rhyming_sentence(hmm64, ids_map, ids_map[rhyme], n_words = 7)
text += '\n'

print(text)
