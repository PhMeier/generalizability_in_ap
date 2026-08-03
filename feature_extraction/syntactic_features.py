import spacy
import stanza
from collections import Counter, defaultdict
from nltk import ngrams, word_tokenize
from sympy.physics.units import amount
from collections import defaultdict
import numpy as np




class SyntaxExtractor():
    def __init__(self):
        self.count_bigrams = Counter()
        self.count_trigrams = Counter()
        self.count_quadgrams = Counter()
        self.depths = {}
        self.nlp = spacy.load("en_core_web_trf")
        self.stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
        self.pos_tag_counts = defaultdict(int)
        # https://downloads.cs.stanford.edu/nlp/software/dependencies_manual.pdf
        self.modifier_relations = {"advmod", "advcl", "npadvmod", "amod", "acl", "relcl", "nmod", "appos", "nummod",
                                   "quantmod", "compound", "poss", "neg"}
        self.adv_dependencies = {"advmod", "advcl", "advcl:relcl", "npadvmod"}

    def mean_number_of_words_per_sentence(self, doc) -> float:
        sentences = list(doc.sents)

        if not sentences:
            return 0.0

        total_words = sum(
            sum(1 for token in sent if not token.is_punct)
            for sent in sentences
        )

        return total_words / len(sentences)


    def sentence_length_range(self, doc) -> int:
        sentence_lengths = [
            sum(1 for token in sent if not token.is_punct)
            for sent in doc.sents
        ]

        if not sentence_lengths:
            return 0

        return max(sentence_lengths) - min(sentence_lengths)

    def get_pos_tag_ngrams(self, data) -> tuple[Counter, Counter, Counter]:
        total_pos_tag_bigrams = []
        total_pos_tag_trigrams = []
        total_pos_tag_quadgrams = []
        for line in data:
            subl = []
            if line:
                doc = self.nlp(line)
                for token in doc:
                    self.pos_tag_counts[token.pos_] = 1 + self.pos_tag_counts.get(token.pos_, 0)
                    subl.append(token.pos_)
                    pos_bigram = list(ngrams(line.split(), 2))
                    pos_trigram = list(ngrams(line.split(), 3))
                    pos_quadgram = list(ngrams(line.split(), 4))
                    total_pos_tag_bigrams.extend(pos_bigram)
                    total_pos_tag_trigrams.extend(pos_trigram)
                    total_pos_tag_quadgrams.extend(pos_quadgram)
        self.count_bigrams = Counter(total_pos_tag_bigrams)
        self.count_trigrams = Counter(total_pos_tag_trigrams)
        self.count_quadgrams = Counter(total_pos_tag_quadgrams)
        return self.count_bigrams, self.count_trigrams, self.count_quadgrams


    def wrap_syntactic_features(self, doc):
        if len(doc)==0:
            return {
                "ramification_factor": 0,
                "tree_width": 0,
                "tree_depth": 0,
                "dependency_relations_per_sentence": defaultdict(lambda: 0),
                "count_modifiers": 0,
                "total_relations":0,
                "freq_adv_dep": 0,
                "mod_verbs": 0,
                "freq_verb_complex_tenses": 0,
                "freq_of_gram_cats": defaultdict(lambda: 0),
                "gram_freqs_common": defaultdict(lambda: 0),
                "freq_comparative_superlative_adj_adv": 0,
                "present_count": 0,
                "past_count": 0,
                "verb_count": 0,
                "ner_count": 0,
                "total_tokens":0
            }


        # dependency
        ramification_factor = self.calc_ramfification_factor(doc) # 0
        tree_width = self.calc_tree_width(doc) # 0
        try:
            tree_depth = self.calc_dependency_parse_depth(doc) # 0
        except RecursionError:
            tree_depth = np.nan
        dependendency_relations_per_sentence = self.dependency_relation_counts(doc) # dict
        # 24.04
        modifier_count, total_relations = self.count_of_modifiers_per_user_tree(doc) # float
        freq_adv_dep = self.frequency_of_adverbial_dependencies(doc) # float
        mod_verbs, verbs = self.frequency_of_modal_verbs(doc) # float
        freq_verb_complex_tenses = self.frequency_of_verbs_in_complex_tenses(doc) # float
        # TODO: Update the len(text)==0 case

        freq_of_gram_cats, gram_freqs_common, total_tokens = self.calc_frequency_of_grammatical_categories(doc) # dict
        freq_comparative_and_superlative_adjectives_and_adverbs = self.calc_freq_comparative_and_superlative_adjectives_and_adverbs(doc) # int
        present_count, past_count = self.calc_rel_frequency_of_present_and_past_tenses(doc) # int

        ner_ratio = self.calc_NER_ratio(doc) #int

        return {
            "ramification_factor": ramification_factor,
            "tree_width": tree_width,
            "tree_depth": tree_depth,
            "dependency_relations_per_sentence": dependendency_relations_per_sentence,
            "count_modifiers": modifier_count,
            "total_relations": total_relations,
            "freq_adv_dep": freq_adv_dep,
            "mod_verbs": mod_verbs,
            "verb_count": verbs,
            "freq_verb_complex_tenses": freq_verb_complex_tenses,
            "freq_of_gram_cats": freq_of_gram_cats,
            "gram_freqs_common": gram_freqs_common,
            "freq_comparative_superlative_adj_adv": freq_comparative_and_superlative_adjectives_and_adverbs,
            "present_count": present_count,
            "past_count": past_count,
            "ner_count": ner_ratio,
            "total_tokens": total_tokens
        }



    def count_of_modifiers_per_user_tree(self, doc) -> float:
        """
        V2: Now per user tree, not in total
        Following the UD English EWT  https://universaldependencies.org/treebanks/en_ewt/index.html and
        https://universaldependencies.org/u/dep/
        Count the modifier relations and calulate the percent: (Modifier attr/total_amount_of_tags)
        - advcl: Averbial clause modifier: The accident *happened* as night was falling. Modifier, which modifies a verb or other predicate (acj. etc)
        - advmod: Adverb or adverbial phrase that modifies a predicate or modifier word, e.g less often.
        - nmod: Nominal depdent on another nominal: a room in the hotel (room --> hotel)
        - appos: Appositional modifier: Nominal that follows the first noun which should be defined, modified, named or described: Sam, my brother
        - nummod: Any number that modifies the meaning of the noun with a quantity: Sam ate 3 steaks.
        - acl: Finite and non-finite clauses that modify a nominal. "The issues as he sees them"
        - amod: Adjectival modifier, adjective thats modifies a noun or pronoun: sam eats large hot dogs.
        - acl:relcl: relative clause modifier: I saw the man you love (man - love)
        - advcl:relcl: Adverbial rel. clause modifier: I tried to expalain myself - which was a bad idea (tried - idea)
        - nmod:desc: Mr. Mago
        - nmod:poss: Possessive nominal modifier: Marie's book
        - nmod:unmarked: Noun phrase as adnominal adverbial modifier:  non-possessive modifier within a nominal takes the form of a nominal lacking a preposition
        I want that color kitten
        - obl:tmod: Last night, I swam in the pool.

        :param text: INput representation of an author
        :return: float, percent of modifiers in the text of an author
        """
        modifier_count = 0
        total_relations = 0

        for token in doc:

            # ignore ROOT if desired
            if token.dep_ != "ROOT":
                total_relations += 1

            if token.dep_ in self.modifier_relations:
                modifier_count += 1

        return modifier_count, total_relations

    def frequency_of_modifiers_per_user(self, doc) -> float:
        """
        Following the UD English EWT  https://universaldependencies.org/treebanks/en_ewt/index.html and
        https://universaldependencies.org/u/dep/
        Count the modifier relations and calulate the percent: (Modifier attr/total_amount_of_tags)
        - advcl: Averbial clause modifier: The accident *happened* as night was falling. Modifier, which modifies a verb or other predicate (acj. etc)
        - advmod: Adverb or adverbial phrase that modifies a predicate or modifier word, e.g less often.
        - nmod: Nominal depdent on another nominal: a room in the hotel (room --> hotel)
        - appos: Appositional modifier: Nominal that follows the first noun which should be defined, modified, named or described: Sam, my brother
        - nummod: Any number that modifies the meaning of the noun with a quantity: Sam ate 3 steaks.
        - acl: Finite and non-finite clauses that modify a nominal. "The issues as he sees them"
        - amod: Adjectival modifier, adjective thats modifies a noun or pronoun: sam eats large hot dogs.
        - acl:relcl: relative clause modifier: I saw the man you love (man - love)
        - advcl:relcl: Adverbial rel. clause modifier: I tried to expalain myself - which was a bad idea (tried - idea)
        - nmod:desc: Mr. Mago
        - nmod:poss: Possessive nominal modifier: Marie's book
        - nmod:unmarked: Noun phrase as adnominal adverbial modifier:  non-possessive modifier within a nominal takes the form of a nominal lacking a preposition
        I want that color kitten
        - obl:tmod: Last night, I swam in the pool.

        :param text: INput representation of an author
        :return: float, percent of modifiers in the text of an author
        """
        counts = {}
        total_doc_size = len(doc)
        for token in doc:
            if token.dep_ in self.modifier_relations:
                counts[token.dep_] = 1 + counts.get(token.dep_, 0)
        freq = (sum(counts.values()) / total_doc_size)
        return freq


    def frequency_of_adverbial_dependencies(self, doc) -> float:
        """
        Calculated on the whole representation.
        Relations, that express circumstancial information about an event, state or predicate (time, manner, reason,
        condition, palce or degree)
        - advmod: less often
        - advcl: She left because it was late
        Not considering obl since it can be argument or modifier
        :param text:
        :return:
        """
        counts = {}
        total_doc_size = len(doc)
        for token in doc:
            #print(token.dep_)
            if token.dep_ in self.adv_dependencies:
                counts[token.dep] = 1 + counts.get(token.dep, 0)
        #freq = (sum(counts.values()) / total_doc_size)
        freq = sum(counts.values())
        #print(percent)
        return freq

    def frequency_of_modal_verbs(self, doc) -> float:
        """
        Calculate the frequency of modal verbs in comparison to the amount of total verbs.
        In Spacy, modal verbs do not have a separate POS category. I was not able to idenfity it through  morophology, e.g
        VerbType=Mod.
        So I used token.dep_ == "aux" and token.tag_ == "MD", this should work for english according to
        https://stackoverflow.com/questions/59713284/use-spacy-models-to-find-modal-verb-for-languages-fr-es-ru
        :param text:
        :return:
        """
        count_of_verbs = 0
        modal_count = 0
        for token in doc:
            if token.pos_ in {"VERB", "AUX"}:
                count_of_verbs += 1
                #print(token.text, token.dep_, token.tag_)
                if token.dep_ == "aux" and token.tag_ == "MD":
                #if "VerbType=Mod" in token.morph:
                    modal_count += 1
        if count_of_verbs > 0:
            return modal_count, count_of_verbs
        else:
            return 0, count_of_verbs

    def frequency_of_verbs_in_complex_tenses(self, doc) -> float:
        """
        Calculated on the whole representation.
        percentage of verbs that appear in complex tenses referred to as “verb chains” (VCs)
        see 3.1 https://aclanthology.org/E17-2108.pdf
        What is a verb chain excatly or complex tenses?
        - She was going
        - We had been going
        "We had been going to the same restaurant for five years. She walks down the street"
        --> Gets a score of 50
        :param text:
        :return:
        """
        AUX_DEPS = {"aux", "auxpass", "aux:pass"}
        verb_count = 0
        verb_in_vc = 0
        for token in doc:
            if token.pos_ == "VERB":
                verb_count += 1
                # check if participant in verb chain
                has_aux = any(child.dep_ in AUX_DEPS for child in token.children)
                if has_aux:
                    verb_in_vc += 1
                #print(token.pos, token.text, list(token.children))
        #percentage = (verb_in_vc / verb_count) if verb_count else 0
        return verb_in_vc

    def dependency_relation_counts(self, doc) -> dict:
        """
        Count in
        Friday 24.04
        Per Sentence --> Count all and divide by the number of sentences? --> Ratio?
        :param doc:
        :return:
        """

        DEPS = ['ROOT', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos', 'attr', 'aux', 'auxpass', 'case', 'cc',
         'ccomp', 'compound', 'conj', 'csubj', 'csubjpass', 'dative', 'dep', 'det', 'dobj', 'expl', 'intj', 'mark',
         'meta', 'neg', 'nmod', 'npadvmod', 'nsubj', 'nsubjpass', 'nummod', 'oprd', 'parataxis', 'pcomp', 'pobj',
         'poss', 'preconj', 'predet', 'prep', 'prt', 'punct', 'quantmod', 'relcl', 'xcomp']

        dependency_relation_counts = {dep: 0 for dep in DEPS}
        sentences = list(doc.sents)
        count_of_sentences = len(sentences)

        if count_of_sentences == 0:
            return {dep: 0 for dep in DEPS}

        for token in doc:
            if token.dep_ in dependency_relation_counts:
                dependency_relation_counts[token.dep_] += 1
        return {
            dep: dependency_relation_counts[dep]/count_of_sentences
            for dep in DEPS
        }


    def calc_ramfification_factor(self, doc) -> float:
        """
        # https://aclanthology.org/E17-2108.pdf
        Calculate the ramification factor for the whole representation
        :param doc: Representation of the author
        :return:
        """
        levels = {}
        for token in doc:
            #print(token.dep_)
            if token.dep not in levels:
                levels[token.dep] = 0 # dep is the integer representation of the dependency relation
            #print(list(token.children))
            levels[token.dep] += len(list(token.children))
        return sum(levels.values())/len(levels) # divide the number of children through the levels


    # added 02/03/26
    def add_ramification_factor(self, sentence):
        if len(sentence) == 0:
            return 0
        levels = {}
        doc = self.nlp(sentence)
        for token in doc:
            #print(token.dep_)
            if token.dep not in levels:
                levels[token.dep] = 0 # dep is the integer representation of the dependency relation
            #print(list(token.children))
            levels[token.dep] += len(list(token.children))
        return sum(levels.values())/len(levels) # divide the number of children through the levels

    def get_tree_width(self, sentence):
        if len(sentence) == 0:
            return 0
        maximum = 0
        doc = self.nlp(sentence)
        for token in doc:
            maximum = max(len(list(token.children)), maximum)
        return maximum


    def calc_tree_width(self, doc) -> int:
        """
        CAlculate the tree width for the whole representation
        :param doc:
        :return:
        """
        maximum = 0
        for token in doc:
            maximum = max(len(list(token.children)), maximum)
        return maximum


    def calc_dependency_parse_depth(self, doc) -> float:
        depth = [self.walk_tree(sent.root, 0) for sent in doc.sents][0]
        return depth


    def get_dependency_parse_depth(self, line) -> float:
        doc = self.nlp(line)
        depth = [self.walk_tree(sent.root, 0) for sent in doc.sents][0]
        return depth


    def walk_tree(self, n, depth) -> float:
        # exclude document which cause a recursion error
        if n.n_lefts + n.n_rights > 0:
            return max(self.walk_tree(child, depth + 1) for child in n.children)
        else:
            return depth




    def calc_tag_ratio(self, sentence, tag) -> float:
        pos_tag_counts = defaultdict(lambda: 0)
        doc = self.nlp(sentence)
        for token in doc:
            pos_tag_counts[token.pos_] = 1 + pos_tag_counts.get(token.pos_, 0)
        tag_ratio = pos_tag_counts.get(tag, 0) / sum(pos_tag_counts.values())
        return tag_ratio

    def calc_frequency_of_grammatical_categories(self, doc) -> dict:
        """
        Returns
                    #(d["ADJ"]/total, d["ADV"]/total, d["INTJ"]/total, d["NOUN"]/total, d["PROPN"]/total, d["VERB"]/total,
            #    d["ADP"]/total, d["AUX"]/total,d["CCONJ"]/total,d["DET"]/total,d["NUM"]/total,d["PART"]/total,
            #    d["PRON"]/total,d["SCONJ"]/total,d["SYM"]/total,d["X"]/total)
        exclude punctuations!
        :param doc:
        :return:

        ".": "punctuation mark,
        sentence closer",
         ",": "punctuation mark,
         comma",
          "-LRB-": "left round bracket",
          "-RRB-": "right round bracket",
          "`": "opening quotation mark",
          '""': "closing quotation mark",
          "''": "closing quotation mark",
           ":": "punctuation mark, colon or ellipsis",
            "$": "symbol, currency",
             "#": "symbol, number sign",
              "AFX": "affix",
               "CC": "conjunction,
                coordinating",
                 "CD": "cardinal number",
                  "DT": "determiner",
                   "EX": "existential there",
                    "FW": "foreign word",
                     "HYPH": "punctuation mark, hyphen",
                      "IN": "conjunction, subordinating or preposition",
                       "JJ": "adjective (English), other noun-modifier (Chinese)",
                        "JJR": "adjective, comparative",
                         "JJS": "adjective, superlative",
                          "LS": "list item marker",
                           "MD": "verb, modal auxiliary",
                            "NIL": "missing tag",
                             "NN": "noun, singular or mass",
                              "NNP": "noun, proper singular",
                               "NNPS": "noun, proper plural",
                                "NNS": "noun, plural",
                                 "PDT": "predeterminer",
                                  "POS": "possessive ending",
                                  "PRP": "pronoun, personal",
                                   "PRP$": "pronoun, possessive",
                                    "RB": "adverb",
                                     "RBR": "adverb,comparative",
                                      "RBS": "adverb, superlative",
                                       "RP": "adverb, particle",
                                        "TO": 'infinitival "to"',
                                         "UH": "interjection",
                                          "VB": "verb, base form",
                                           "VBD": "verb, past tense",
                                            "VBG": "verb, gerund or present participle",
                                             "VBN": "verb, past participle",
                                              "VBP": "verb, non-3rd person singular present",
                                               "VBZ": "verb, 3rd person singular present",
                                                "WDT": "wh-determiner",
                                                 "WP": "wh-pronoun, personal",
                                                  "WP$": "wh-pronoun, possessive",
                                                  "WRB": "wh-adverb",
                                                  "SP": "space (English), sentence-final particle (Chinese)",
                                                   "ADD": "email", "NFP": "superfluous punctuation",
                                                    "GW": "additional word in multi-word expression",
                                                    "XX": "unknown",
                                                    "BES": 'auxiliary "be"',
                                                     "HVS": 'forms of "have"',
                                                      "_SP": "whitespace",




        """
        # See here:
        # https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
        # and https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
        # took the onotnotes 5/Penn Treebank tagset:     # OntoNotes 5 / Penn Treebank
        #http://www.snlp.de/prescher/teaching/2007/Parsing/bib/2007parsing.georgieva.dworaczek.pdf
        TAGS = [
            ".",
            ",",
            "-LRB-",
            "-RRB-",
            "``",
            '""',
            "''",
            ":",
            "$",
            "#",
            "AFX",
            "CC",
            "CD",
            "DT",
            "EX",
            "FW",
            "HYPH",
            "IN",
            "JJ",
            "JJR",
            "JJS",
            "LS",
            "MD",
            "NIL",
            "NN",
            "NNP",
            "NNPS",
            "NNS",
            "PDT",
            "POS",
            "PRP",
            "PRP$",
            "RB",
            "RBR",
            "RBS",
            "RP",
            "TO",
            "UH",
            "VB",
            "VBD",
            "VBG",
            "VBN",
            "VBP",
            "VBZ",
            "WDT",
            "WP",
            "WP$",
            "WRB",
            "SP",
            "ADD",
            "NFP",
            "GW",
            "XX",
            "BES",
            "HVS",
            "_SP",
        ]
        GRAMMATICAL_CATEGORY_MAP = {
            "PUNCT":[".", ",",":", "HYPH"],
            "BRACKETS": ["-LRB-", "-RRB-"],
            "QUOTATION": ["``", '""', "''"],
            "SYMBOL": ["$","#"],
            "ADJECTIVE": ["JJ", "JJR", "JJS"],
            "COMMON_NOUN": ["NN", "NNS"],
            "PROPER_NOUN": ["NNP", "NNPS"],
            "ADVERB": ["RB", "RBR", "RBS", "RP"],
            "VERB": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
            "WH": ["WDT", "WP", "WP$", "WRB"],
            "DETERMINER": ["DT", "PDT"],
            "PRONOUN": ["PRP", "PRP$"],
            "CONJUNCTION_ADPOSITION": ["IN", "CC"],
            "SPACE": ["SP", "_SP"],
            "AUX_SPECIAL": ["BES", "HVS"]
        }



        d = defaultdict(lambda: 0)
        tokens = [t for t in doc] #if not t.is_space] # exclude punctuation
        total = len(tokens)
        if total == 0:
            return {tag: 0. for tag in TAGS}, {tag: 0. for tag in GRAMMATICAL_CATEGORY_MAP}
        for token in tokens:
            d[token.tag_] += 1
        gram_freqs = {category: sum(d.get(tag, 0.0) for tag in tags)#/total
            for category, tags in GRAMMATICAL_CATEGORY_MAP.items()
        }
        return {tag: d[tag] for tag in TAGS}, gram_freqs, total


    def calc_freq_comparative_and_superlative_adjectives_and_adverbs(self, doc) -> float:
        """
        # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        Penn TreeBank tag of comparative adjective: JJR
        Penn TreeBank tag of superlative adjective: JJS
        Penn TreeBank Tag of comparative adverb: RBR
        Penn TreeBank Tag of superlative adverb: RBS
        :param text:
        :return:
        """
        tags = {"JJR", "JJS", "RBR", "RBS"}
        amount_of_tokens = len(doc)
        count = 0
        for token in doc:
            #print(token.text, token.pos_, token.tag_)
            if token.tag_ in tags:
                count += 1
        return count #(count/amount_of_tokens)


    def calc_rel_frequency_of_present_and_past_tenses(self, doc) -> tuple:
        """
        Use mophological information to retrieve present and past
        # running: VerbForm = PART
        - to go: VerbForm INF
        :param text:
        :return:
        """
        verbs = {"VERB", "AUX"}
        present_count = 0
        past_count = 0
        total_verbs = 0
        for token in doc:
            if token.pos_ in verbs:
                total_verbs += 1
                tense = token.morph.get("Tense")
                if "Pres" in tense:
                    present_count += 1
                if "Past" in tense:
                    past_count += 1
        return present_count, past_count
            #print(token.text, token.morph)



    def calc_NER_ratio(self, doc) -> float:
        if doc.ents:
            return len(doc.ents) #/len(doc)
        else:
            return 0.


