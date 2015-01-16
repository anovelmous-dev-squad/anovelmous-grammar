from nltk import data, Text
from nltk.corpus import brown
import numpy as np
import string
import os
import json


class GrammarFilter(object):
    """
    An object used to filter out all uncommon word sequences in a given chapter.
    """

    def __init__(self, vocabulary, corpus=None):
        self.vocabulary = vocabulary
        self.vocabulary_lookup = {token: True for token in self.vocabulary}
        self.tokenizer = data.load('tokenizers/punkt/english.pickle')

        corpora_cache_fp = os.path.join(os.path.dirname(__file__), 'corpora_cache')
        if not os.path.exists(corpora_cache_fp):
            os.makedirs(corpora_cache_fp)

        full_brown_corpus_file_path = os.path.join(corpora_cache_fp, 'full_brown_corpus.npy')
        full_brown_bigrams_file_path = os.path.join(corpora_cache_fp, 'full_brown_bigrams.json')

        if corpus:
            self.corpus = corpus
            self.bigrams = self.build_vocab_targeted_bigrams()
        elif not corpus \
                and os.path.exists(full_brown_corpus_file_path) \
                and os.path.exists(full_brown_bigrams_file_path):
            self.corpus = np.load(full_brown_corpus_file_path)
            with open(full_brown_bigrams_file_path) as f:
                self.bigrams = json.load(f)
        else:
            brown_text = Text(word.lower() for word in brown.words())
            self.corpus = np.array(brown_text.tokens)
            self.bigrams = self.build_vocab_targeted_bigrams()
            np.save(full_brown_corpus_file_path, self.corpus)
            with open(full_brown_bigrams_file_path, 'w') as f:
                json.dump(self.bigrams, f)

    def build_vocab_targeted_bigrams(self):
        vocab_occurrences = {vocab_term: {} for vocab_term in self.vocabulary}

        preceding_token = self.corpus[0]
        encountered_punctuation = False
        for token in self.corpus[1:]:
            if token in string.punctuation:
                encountered_punctuation = True
                continue

            if encountered_punctuation:
                preceding_token = token
                encountered_punctuation = False
                continue

            if self.vocabulary_lookup.get(token):
                vocab_occurrences[token][preceding_token] = True

            preceding_token = token

        return vocab_occurrences

    def is_occurring_combination(self, candidate_token, preceding_token):
        return self.bigrams[candidate_token].get(preceding_token)

    def get_grammatically_correct_vocabulary_subset(self, preceding_token):
        """
        Returns a subset of a given vocabulary based on whether its terms are "grammatically correct". This status
        is defined by whether the bigram has ever occured in the corpus.
        """
        return [token for token in self.vocabulary
                if self.is_occurring_combination(token, preceding_token)]
