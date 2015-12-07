from nltk import data, Text, word_tokenize, pos_tag
from nltk.corpus import brown
import numpy as np
from scipy import stats
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
        full_brown_token_sequences_fp = os.path.join(
            corpora_cache_fp, 'full_brown_token_sequences.json'
        )

        if corpus:
            self.corpus = corpus
            self.bigrams = self.build_vocab_targeted_bigrams()
            self.trigrams = self.build_vocab_targeted_trigrams()
            self.pos_sequences, self.token_sequences = \
                self.build_pos_sequences_and_token_sequences()
        elif not corpus \
                and os.path.exists(full_brown_corpus_fp) \
                and os.path.exists(full_brown_bigrams_fp) \
                and os.path.exists(full_brown_trigrams_fp) \
                and os.path.exists(full_brown_pos_sequences_fp) \
                and os.path.exists(full_brown_token_sequences_fp):
            self.corpus = np.load(full_brown_corpus_fp)
            with open(full_brown_bigrams_fp) as f:
                self.bigrams = json.load(f)
            with open(full_brown_trigrams_fp) as f:
                self.trigrams = json.load(f)
            with open(full_brown_pos_sequences_fp) as f:
                self.pos_sequences = json.load(f)
            with open(full_brown_token_sequences_fp) as f:
                self.token_sequences = json.load(f)
        else:
            brown_text = Text(word.lower() for word in brown.words())
            self.corpus = np.array(brown_text.tokens)
            self.bigrams = self.build_vocab_targeted_bigrams()
            self.trigrams = self.build_vocab_targeted_trigrams()
            self.pos_sequences, self.token_sequences = \
                self.build_pos_sequences_and_token_sequences()
            np.save(full_brown_corpus_fp, self.corpus)
            with open(full_brown_bigrams_fp, 'w') as f:
                json.dump(self.bigrams, f)
            with open(full_brown_trigrams_fp, 'w') as f:
                json.dump(self.trigrams, f)
            with open(full_brown_pos_sequences_fp, 'w') as f:
                json.dump(self.pos_sequences, f)
            with open(full_brown_token_sequences_fp, 'w') as f:
                json.dump(self.token_sequences, f)

    def build_pos_sequences_and_token_sequences(self):
        pos_sequences = {}
        token_sequences = {}
        for sent in brown.tagged_sents():
            positional_dict = pos_sequences
            trigrams = [sent[i: i+3] for i in range(len(sent) - 2)]
            for trigram in trigrams:
                token_sequence = ' '.join([token[0] for token in trigram])
                tag_sequence = [token[1] for token in trigram]
                token_sequences[token_sequence] = tag_sequence
                for i, (token, tag) in enumerate(trigram):
                    end_term = True if i == 2 else False
                    if end_term:
                        if positional_dict.get(tag):
                            positional_dict[tag] += 1
                        else:
                            positional_dict[tag] = 1
                        continue

                    if not positional_dict.get(tag):
                        positional_dict[tag] = {}

                    positional_dict = positional_dict[tag]
                positional_dict = pos_sequences
        for first_token, subsequent_options in pos_sequences.items():
            for second_token, final_options in subsequent_options.items():
                filtered_options = {}
                if len(final_options) < 2:
                    filtered_options = {k: 1.0 for k in final_options.keys()}
                else:
                    filtered_options = self.filter_by_min_confidence_interval(
                        final_options,
                        confidence=0.95
                    )
                pos_sequences[first_token][second_token] = filtered_options
        return pos_sequences, token_sequences

    def filter_by_min_confidence_interval(self, pos_options, confidence=0.95):
        total_occurences = sum(pos_options.values())
        pos_means = {k: v / total_occurences for k, v in pos_options.items()}
        bayes_results = stats.bayes_mvs(list(pos_means.values()), confidence)
        threshold = bayes_results[2].minmax[0]
        return {k: v for k, v in pos_means.items() if v > threshold}

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

    def get_likelihood(self, prev2_token, prev_token, new_token):
        trigram = [prev2_token, prev2_token, new_token]
        trigram_string = ' '.join(trigram)
        if not self.token_sequences.get(trigram_string):
            return None

        tags = self.token_sequences[trigram_string]
        tagged_tokens = [(token, tags[i])
                         for i, token in enumerate(trigram)]
        positional_dict = self.pos_sequences
        for i, (token, tag) in enumerate(tagged_tokens):
            positional_dict = positional_dict.get(tag, None)
            if positional_dict is None:
                return None

            if i == 2:
                return positional_dict

    def get_grammatically_correct_vocabulary_subset(self, sent,
                                                    sent_filter='combined'):
        """
        Returns a subset of a given vocabulary based on whether its
        terms are "grammatically correct".
        """
        if sent == '':
            return self.vocabulary

        sent_tokens = word_tokenize(sent)

        if sent_filter == 'combined':
            combined_filters = self.get_pos_filtered_vocab(sent_tokens) + \
                               self.get_trigram_filtered_vocab(sent_tokens) + \
                               self.get_bigram_filtered_vocab(sent_tokens)
            return combined_filters

        if sent_filter == 'pos' and len(sent_tokens) > 1:
            return self.get_pos_filtered_vocab(sent_tokens)
        elif sent_filter == 'bigram' or len(sent_tokens) < 2:
            return self.get_bigram_filtered_vocab(sent_tokens)
        elif sent_filter == 'trigram':
            return self.get_trigram_filtered_vocab(sent_tokens)

    def get_pos_filtered_vocab(self, sent_tokens):
        token_likelihoods = self.get_likelihood_subset_by_pos_filter(
            prev2_token=sent_tokens[-2], prev_token=sent_tokens[-1]
        )
        sorted_tokens = sorted(token_likelihoods, key=lambda t: t[1])
        return [token for token, likelihood in sorted_tokens]

    def get_trigram_filtered_vocab(self, sent_tokens):
        preceding_token = sent_tokens[-1]
        return self.get_subset_by_bigram_filter(preceding_token)

    def get_bigram_filtered_vocab(self, sent_tokens):
            prev2_token = sent_tokens[-2]
            prev_token = sent_tokens[-1]
            return self.get_subset_by_trigram_filter(prev2_token, prev_token)

    def get_likelihood_subset_by_pos_filter(self, prev2_token, prev_token):
        tokens = []
        for token in self.vocabulary:
            likelihood = self.get_likelihood(prev2_token, prev_token, token)
            if not likelihood:
                continue
            tokens.append((token, likelihood))
        return tokens

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
