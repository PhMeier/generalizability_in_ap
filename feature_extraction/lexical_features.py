import nltk
import pandas as pd
from nltk import ngrams
import numpy as np
from collections import Counter
import spacy
import textstat
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from lexicalrichness import LexicalRichness
from spellchecker import SpellChecker
from nltk.corpus import stopwords

"""
Taken from NewOrder
"""


class LexicalFeatures():
    def __init__(self):
        self.n_gram_counts = Counter()
        self.words = Counter()
        self.nlp = spacy.load("en_core_web_trf")
        self.pos_tag_counts = {}
        self.n_gram_counts = Counter()
        self.spellchecker = SpellChecker()
        self.stopwords = stopwords.words("english")

    def get_ngrams_corpus(self, data, n):
        """
        Expects a list of string to compute n grams
        :param data:
        :param n:
        :return:
        """
        total_ngrams = []
        for line in data:
            if line:
                ngram = list(ngrams(line.split(), n))
                # print(list(ngram))
                total_ngrams.extend(ngram)
        # print(total_ngrams)
        self.n_gram_counts = Counter(total_ngrams)
        return self.n_gram_counts

    def get_ngrams_sentence(self, sentence, n):
        """
        Expects a list of string to compute n grams
        :param data:
        :param n:
        :return:
        """
        total_ngrams = []
        if sentence:
            ngram = list(ngrams(sentence.split(), n))
            total_ngrams.extend(ngram)
        self.n_gram_counts = Counter(total_ngrams)
        return self.n_gram_counts

    def count_words(self, data) -> Counter:
        """
        Counts the words given a list of documents
        :param data: list of strings
        :return words: Counter object
        """
        stop_words = stopwords.words('english')
        stopwords_dict = Counter(stop_words)
        for line in data:
            if line:
                # print(line)
                # remove stopwords
                line = line.lower()
                line = " ".join([word for word in line.split() if word not in stopwords_dict])
                tw_words = line.split()
                self.words.update(tw_words)
        return self.words

    def get_average_word_length_single_instance(self, all_texts) -> tuple[float, float, int, int]:
        """
        Return the average word length, maximum and minum word length for a single instance which can be one sentence or more
        :param data:
        :return:
        """
        if len(all_texts) == 0:
            return 0, 0, 0, 0
        words = [len(i) for i in all_texts.split()]
        maximum = max(words)
        minimum = min(words)
        w_length = sum(words) / len(words)
        std_w_length = np.std(words)
        return w_length, std_w_length, maximum, minimum

    def get_average_word_length(self, data) -> tuple[float, int, int]:
        """
        Return the average word length, maximum and minum word length
        :param data:
        :return:
        """
        w_length = 0
        maximum = 0
        minimum = 10000
        for line in data:
            if line:
                words = [len(i) for i in line.split()]
                maximum = max(maximum, max(words))
                minimum = min(minimum, min(words))
                w_length += sum(words) / len(words)
        w_length = w_length / len(data)
        return w_length, maximum, minimum

    def get_word_ngrams_corpus(self, data, n) -> Counter:
        """
        Returns the N-Gram counts of words.
        :param data:
        :param n: Defines whether to produce uni, bi, tri grams etc.
        :return self.n_gram_counts: Counter object
        """
        total_ngrams = []
        for line in data:
            if line:
                ngram = list(ngrams(line.split(), n))
                # print(list(ngram))
                total_ngrams.extend(ngram)
        # print(total_ngrams)
        self.n_gram_counts = Counter(total_ngrams)
        return self.n_gram_counts

    def get_word_ngrams(self, sentence, n) -> Counter:
        """
        Returns the N-Gram counts of words.
        :param data:
        :param n: Defines whether to produce uni, bi, tri grams etc.
        :return self.n_gram_counts: Counter object
        """
        total_ngrams = []
        ngram = list(ngrams(sentence.split(), n))
        total_ngrams.extend(ngram)
        # print(total_ngrams)
        self.n_gram_counts = Counter(total_ngrams)
        return self.n_gram_counts

    def get_lexical_diversity_sentence_level(self, sentence):
        """
        Calculates type token ratio and corrected type token ratio on a single sentence.
        If you have multiple sentences to evaluate, use get_lexical_diversity_corpus
        :param sentence:
        :return:
        """
        if sentence:
            lex = LexicalRichness(sentence)
            try:
                return lex.ttr, lex.cttr
            except ZeroDivisionError:
                return 0., 0.
        else:
            return 0., 0.

    def get_lexical_diversity_corpus(self, data):
        """
        Calculates type token ratio and corrected type token ratio on a given corpus
        :param data:
        :return:
        """
        joined_data = " ".join(data)
        if joined_data:
            lex = LexicalRichness(joined_data)
            try:
                return lex.ttr, lex.cttr
            except ZeroDivisionError:
                return 0., 0.
        else:
            return 0., 0.

    def count_capital_and_lower_cases_corpus(self, data):
        for text in data:
            sent_text = nltk.sent_tokenize(text)
            capital_ratio = 0.
            first_letter_capitalized = 0.
            capitalized_without_first_word = 0.
            for sentence in sent_text:
                words = sentence.split()
                all_letters = sum(c.isalpha() for c in sentence)
                all_capital = sum(c.isupper() for c in sentence)
                # capital_words = sum(1 for word in words if word.isupper() and len(word) > 1)

                capital_ratio_sentence = all_capital / all_letters if all_letters > 0 else 0
                list_first_letter = words[1:]
                capitalized_without_first_word_sentence = sum(1 for word in list_first_letter if word[0].isupper())
                first_letter_capitalized_sentence = sum(1 for word in words if word[0].isupper())

                capital_ratio += capital_ratio_sentence
                first_letter_capitalized += first_letter_capitalized_sentence
                capitalized_without_first_word += capitalized_without_first_word_sentence
        return capital_ratio, first_letter_capitalized, capitalized_without_first_word

    def spelling_errors(self, sentence):
        words = sentence.split()
        spelling_errors = self.spellchecker.unknown(words)
        return len(spelling_errors)

    def spelling_errors_corpus(self, data):
        spell_errors = 0
        for sentence in data:
            words = sentence.split()
            spell_errors += len(self.spellchecker.unknown(words))
        return spell_errors, spell_errors / len(data)

    def get_reading_easiness(self, sentence):
        ease = textstat.flesch_reading_ease(
            sentence)  # 206.835-1.015(total_words/total_sentences)-84.6(total_syll/total_words)
        grad = textstat.flesch_kincaid_grade(sentence)  #
        dale = textstat.dale_chall_readability_score(
            sentence)  # uses difficult words: 0.1579((diff_words/words)*100) + 0.0496(words/Sentences)
        ari = textstat.automated_readability_index(sentence)
        # ADDED 07.04
        gunning_fog = textstat.gunning_fog(sentence)  # (0.4(words/sentences) + 100(complex_Words/words))
        return ease, grad, dale, ari, gunning_fog

    def get_reading_easiness_corpus(self, data):
        easiness = 0
        grade = 0
        dale = 0
        ari = 0
        corpus = " ".join(data)
        # for sentence in data: # TODO: Korpus basiert! Nicht einzeln --> Satzlänge DONE
        easiness += textstat.flesch_reading_ease(corpus)
        grade += textstat.flesch_kincaid_grade(corpus)
        dale += textstat.dale_chall_readability_score(corpus)
        ari += textstat.automated_readability_index(corpus)
        return easiness, grade, dale, ari

    def wrapper_for_spacy(self, doc) -> [[float, float], float, float, float, float]:
        if len(doc) == 0:
            return (0, 0), 0, 0,0
        #doc = self.nlp(text)
        #mean_number_of_chars_per_words = self.get_mean_number_of_chars_per_word(doc)
        words_with_two_or_three_chars = self.words_with_two_or_three_chars(doc)
        #standard_deviation_of_word_length = self.standard_deviation_of_word_length(doc)
        #diff_longest_and_shortest_word = self.diff_longest_and_shortest_word(doc)
        count_stopwords = self.count_stopwords(doc)
        count_acronyms = self.count_acronyms(doc)
        count_first_person_pronouns = self.ratio_first_person_pronouns(doc)
        return (words_with_two_or_three_chars,
                count_stopwords, count_acronyms, count_first_person_pronouns)

    def get_mean_number_of_chars_per_word(self, doc):
        words = [t for t in doc if t.is_alpha]
        if not words:
            return 0,0

        total_chars = sum(len(t.text) for t in words)
        return total_chars, len(words)

    def standard_deviation_of_word_length(self, doc) -> float:
        words = [len(t.text) for t in doc if t.is_alpha]
        std = np.std(words)
        return std

    def diff_longest_and_shortest_word(self, doc):
        words = [len(t.text) for t in doc if t.is_alpha]
        maximum = max(words)
        minimum = min(words)
        return abs(minimum - maximum)

    def count_stopwords(self, doc):
        """
        Sapcy uses its own list of stopwords
        :param doc:
        :return:
        """
        tokens = [t for t in doc if not t.is_space and not t.is_punct]
        stopwords = sum(1 for t in tokens if t.is_stop)
        return stopwords

    def count_acronyms(self, doc):
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        acronyms = sum(1 for t in tokens if t.text.isupper() and t.is_alpha and len(t.text) > 1)
        return acronyms

    def ratio_first_person_pronouns(self, doc):
        FIRST_PERSON = {
            "i", "me", "my", "mine", "myself",
            "we", "us", "our", "ours", "ourselves"
        }
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        count = sum(
            1 for t in tokens
            if t.pos_ == "PRON" and t.lemma_.lower() in FIRST_PERSON  # Lemma for ours --> our
        )
        return count

    def words_with_two_or_three_chars(self, doc) -> [int, int]:

        len_two = 0
        len_three = 0

        for token in doc:
            if token.is_alpha:
                length = len(token.text)
                if length == 2:
                    len_two += 1
                elif length == 3:
                    len_three += 1
        return len_two, len_three
