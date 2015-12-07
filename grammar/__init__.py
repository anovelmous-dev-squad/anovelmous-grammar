from nltk import data, Text, word_tokenize, pos_tag
from nltk.corpus import brown
import numpy as np
import string
import os
import json


class GrammarFilter(object):
    """
    An object used to filter out all uncommon word sequences in a given chapter
    """

    def __init__(self, vocabulary, corpus=None, nltk_data_path=None):
        """

        :param vocabulary: a list of strings to filter by context
        :param corpus: provide your own nltk corpus
        :param nltk_data_path: absolute path to look for the nltk data
                               directory where the corpus is stored.
        """
        self.vocabulary = vocabulary
        self.vocabulary_lookup = {token: True for token in self.vocabulary}

        if nltk_data_path:
            data.path.append(nltk_data_path)
        self.tokenizer = data.load('tokenizers/punkt/english.pickle')

        corpora_cache_fp = os.path.join(
            os.path.dirname(__file__), 'corpora_cache'
        )
        if not os.path.exists(corpora_cache_fp):
            os.makedirs(corpora_cache_fp)

        full_brown_corpus_fp = os.path.join(
            corpora_cache_fp, 'full_brown_corpus.npy'
        )
        full_brown_bigrams_fp = os.path.join(
            corpora_cache_fp, 'full_brown_bigrams.json'
        )
        full_brown_trigrams_fp = os.path.join(
            corpora_cache_fp, 'full_brown_trigrams.json'
        )
        full_brown_pos_sequences_fp = os.path.join(
            corpora_cache_fp, 'full_brown_pos_sequences.json'
        )

        if corpus:
            self.corpus = corpus
            self.bigrams = self.build_vocab_targeted_bigrams()
            self.trigrams = self.build_vocab_targeted_trigrams()
            self.pos_sequences = self.build_pos_sequences_db()
        elif not corpus \
                and os.path.exists(full_brown_corpus_fp) \
                and os.path.exists(full_brown_bigrams_fp) \
                and os.path.exists(full_brown_trigrams_fp) \
                and os.path.exists(full_brown_pos_sequences_fp):
            self.corpus = np.load(full_brown_corpus_fp)
            with open(full_brown_bigrams_fp) as f:
                self.bigrams = json.load(f)
            with open(full_brown_trigrams_fp) as f:
                self.trigrams = json.load(f)
            with open(full_brown_pos_sequences_fp) as f:
                self.pos_sequences = json.load(f)
        else:
            brown_text = Text(word.lower() for word in brown.words())
            self.corpus = np.array(brown_text.tokens)
            self.bigrams = self.build_vocab_targeted_bigrams()
            self.trigrams = self.build_vocab_targeted_trigrams()
            self.pos_sequences = self.build_pos_sequences_db()
            np.save(full_brown_corpus_fp, self.corpus)
            with open(full_brown_bigrams_fp, 'w') as f:
                json.dump(self.bigrams, f)
            with open(full_brown_trigrams_fp, 'w') as f:
                json.dump(self.trigrams, f)
            with open(full_brown_pos_sequences_fp, 'w') as f:
                json.dump(self.pos_sequences, f)

    def build_pos_sequences_db(self):
        pos_sequences = {}
        for sent in brown.tagged_sents():
            positional_dict = pos_sequences
            for token, tag in sent:
                if not positional_dict.get(tag):
                    positional_dict[tag] = {}

                positional_dict = positional_dict[tag]
        return pos_sequences

    def build_vocab_targeted_bigrams(self):
        vocab_occurrences = {vocab_term: {} for vocab_term in self.vocabulary}

        preceding_token = self.corpus[0]
        encountered_punctuation = False
        for token in self.corpus[1:]:
            if token in string.punctuation:
                encountered_punctuation = True
                continue

            if encountered_punctuation:
                encountered_punctuation = False
            elif self.vocabulary_lookup.get(token):
                vocab_occurrences[token][preceding_token] = True

            preceding_token = token

        return vocab_occurrences

    def build_vocab_targeted_trigrams(self):
        vocab_occurrences = {vocab_term: {} for vocab_term in self.vocabulary}

        prev2_token = self.corpus[0]
        prev_token = self.corpus[1]
        encountered_punctuation = False
        for token in self.corpus[2:]:
            if token in string.punctuation:
                encountered_punctuation = True
                continue

            if encountered_punctuation:
                encountered_punctuation = False
            elif self.vocabulary_lookup.get(token):
                vocab_occurrences[token][prev2_token + ' ' + prev_token] = True

            prev2_token = prev_token
            prev_token = token

        return vocab_occurrences

    def is_occurring_bigram(self, preceding_token, candidate_token):
        return self.bigrams[candidate_token].get(preceding_token)

    def is_occurring_trigram(self, prev2_token, prev_token, token):
        return self.trigrams[token].get(prev2_token + ' ' + prev_token)

    def get_grammatically_correct_vocabulary_subset(self, sent,
                                                    sent_filter='pos'):
        """
        Returns a subset of a given vocabulary based on whether its
        terms are "grammatically correct".
        """
        if sent == '':
            return self.vocabulary

        sent_tokens = word_tokenize(sent)

        if sent_filter == 'pos':
            return self.get_subset_by_pos_filter(sent_tokens)
        elif sent_filter == 'bigram' or len(sent_tokens) < 2:
            preceding_token = sent_tokens[-1]
            return self.get_subset_by_bigram_filter(preceding_token)
        elif sent_filter == 'trigram':
            prev2_token = sent_tokens[-2]
            prev_token = sent_tokens[-1]
            return self.get_subset_by_trigram_filter(prev2_token, prev_token)

    def get_subset_by_pos_filter(self, sent_tokens):
        return [token for token in self.vocabulary
                if self.is_occurring_pos_sequence(sent_tokens, token)]

    def get_subset_by_bigram_filter(self, preceding_token):
        if preceding_token in string.punctuation:
            return self.vocabulary

        return [token for token in self.vocabulary
                if self.is_occurring_bigram(preceding_token, token)]

    def get_subset_by_trigram_filter(self, prev2_token, prev_token):
        if prev_token in string.punctuation:
            return self.vocabulary

        return [token for token in self.vocabulary
                if self.is_occurring_trigram(prev2_token, prev_token, token)]

    def is_occurring_pos_sequence(self, sent_tokens, new_token):
        candidate_sent_tokens = sent_tokens[:]
        candidate_sent_tokens.append(new_token)
        tagged_tokens = pos_tag(candidate_sent_tokens)

        positional_dict = self.pos_sequences
        for token, tag in tagged_tokens:
            positional_dict = positional_dict.get(tag, None)
            if positional_dict is None:
                return False

        return True
