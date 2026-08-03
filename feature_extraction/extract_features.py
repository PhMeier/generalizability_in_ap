import ast
import os
import sys
import csv
import pandas as pd
import spacy
import numpy as np
from collections import Counter, defaultdict

current_file = os.path.abspath(__file__)
mother_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.insert(0, mother_dir)
nlp = spacy.load("en_core_web_trf")

import lexical_features
import syntactic_features
import character_features
import discourse_features
import dictionary_features
"""
Input file must be authorid - list of documents of this author
aut1 - text1
aut1 - text2
aut1 - text3
etc
"""

def aggegreate_dataframe(df):
    grouped = (
        df.groupby("author_id")
        .agg({
            "document":list
        })
        .reset_index()
    )
    return grouped


GRAMMATICAL_CATEGORY_MAP = {
    "noun": ["NOUN", "PROPN"],
    "verb": ["VERB", "AUX"],
    "adjective": ["ADJ"],
    "adverb": ["ADV"],
    "pronoun": ["PRON"],
    "determiner": ["DET"],
    "preposition": ["ADP"],
    "conjunction": ["CCONJ", "SCONJ"],
    "numeral": ["NUM"],
    "interjection": ["INTJ"],
    "particle": ["PART"],
    "symbol": ["SYM"],
    "other": ["X"],
}


lf = lexical_features.LexicalFeatures()
sf = syntactic_features.SyntaxExtractor()
cf = character_features.CharacterFeatures()
discf = discourse_features.DiscourseFeatures()
dictio = dictionary_features.DictionaryFeatures()

def average_dictionary_vals_mean(dictionary):
    """
    Takes in a defaultdict(list)
    :param dictionary:
    :return:
    """
    for key,value in dictionary.items():
        dictionary[key] = np.mean(value)
    return dictionary


def average_dictionary_vals_by_count(dictionary, n):
    """
    Takes in a defaultdict(list)
    :param dictionary:
    :return:
    """
    for key,value in dictionary.items():
        dictionary[key] = sum(value) / n if n > 0 else 0
    return dictionary


def add_values_to_dict(dictionary_big, dictionary_new:dict):
    """
    Update the defaultdict dictioanry big with values from a dictionary (dictionary_new)
    :param dictionary_big:
    :param dictionary_new:
    :return:
    """
    for key, value in dictionary_new.items():
        dictionary_big[key].append(value)
    return dictionary_big

def save_division(numerator, denominator):
    try:
        return numerator/denominator
    except ZeroDivisionError:
        return 0.0

def retrieve_maximum_safely(list):
    try:
        return max(list)
    except ValueError:
        return 0.

def retrieve_minimum_safely(list):
    try:
        return min(list)
    except ValueError:
        return 0.

def save_std(list):
    values = np.asarray(list, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) < 2:
        return 0

    return np.std(values)


def feature_extraction(row: list):
    """
    Input is a list of texts of one author. Calculate features for every text and then aggreate.
    Otherwise, Spacy won't execute for some cases
    :param row:
    :return:
    """
    count_of_documents = len(row)
    flesch_reading_easiness_sum, flesch_grade_sum, dale_chall_sum, ari_sum, gf_sum = 0,0,0,0,0
    spelling_errors_sum = 0
    (ramification_factor_list, tree_width_list, tree_depth_list, freq_modifiers_sum,
     freq_adv_dep_sum, freq_mod_verbs_sum, freq_verb_complex_tenses_sum,
     freq_comparative_and_superlative_adjectives_and_adverbs_sum,
     present_sum, past_ratio_sum, ner_count) = [],[],[],0,0,0,0,0,0,0,0
    dependendency_relations_per_sentence_sum = defaultdict(list) #Counter()
    freq_of_gram_cats_sum = defaultdict(list) #Counter()


    freq_gram_freqs_common_sum = defaultdict(list)



    edu_freq_sum = defaultdict(list) # dict
    edu_freq_nuc_sum = defaultdict(list)
    nuclearity_freq_sum = defaultdict(list)

    # if we wanna use ttr and cttr
    all_texts = " ".join(row)
    print("all_texts")
    print(all_texts)

    all_texts_lower = all_texts.lower()
    disc = dictio.count_discourse_markers(all_texts_lower)
    abb = dictio.count_abb(all_texts_lower)
    pos = dictio.count_pos(all_texts_lower)
    neg = dictio.count_neg(all_texts_lower)
    prof = dictio.count_profanity(all_texts_lower)
    interjections = dictio.count_interjections_markers(all_texts_lower)

    # char features
    total_number_of_chars = cf.total_number_of_chars(all_texts) # done
    upper_ratio, lower_ratio = cf.ratio_of_upper_case_and_lower_case_chars(all_texts)
    ratio_of_periods = cf.ratio_of_periods(all_texts)
    ratio_of_commas = cf.ratio_of_commas(all_texts)
    ratio_of_semicolons = cf.ratio_of_semicolons(all_texts)
    ratio_of_colons = cf.ratio_of_colons(all_texts)
    ratio_of_exclamation = cf.ratio_of_exclamation(all_texts)
    ratio_of_parentheses = cf.ratio_of_parentheses(all_texts)
    ratio_of_numbers = cf.ratio_of_numbers(all_texts)
    ratio_of_hyphens = cf.ratio_of_hyphens(all_texts)
    ratio_of_quotation_marks = cf.ratio_of_quotation_marks(all_texts)


    avg_w_length, std_w_length, maximum_w_length, minimum_w_length = lf.get_average_word_length_single_instance(all_texts)  #
    ttr, cttr = lf.get_lexical_diversity_sentence_level(all_texts) # must be calculated on all texts of an author, since it is a corpus metric


    diff_between_min_and_max = []
    imbalance_root_list = []
    total_imbalance_list = []
    edu_freq_list = []

    sentence_count = 0
    total_tokens = 0
    sentence_lengths = []

    total_verbs = 0
    total_present = 0
    total_past = 0



    total_words = 0


    words_with_len_two = 0
    words_with_len_three = 0

    count_stopwords_sum = 0
    count_acronyms_sum = 0
    count_first_pronoun_sum = 0

    total_relations_sum = 0
    total_tokens_sum = 0

    edu_freq_count_total = 0
    edu_freq_nuc_total = 0
    nuclearity_freq_count_total = 0

    std_words_per_sentence = []
    difference_between_longest_and_shortest_word = [] # collect all and take the maximum later
    sentence_length_difference_list = []
    length_of_words_list = []

    for i in range(len(row)): # row is a list of documents written by the author        print("ROW")
        text = row[i]
        print(text)
        if not isinstance(text, str):
            text = ""
        try:
            doc = nlp(text) # spacy nlp pipelinepint
        except ValueError:
            text= ""
            doc = nlp(text) # spacy nlp pipelinepin
        print(row[i])
        sentences = list(doc.sents) # retuns [] when empty
        sentence_count += len(sentences)
        total_words += sum(sum(1 for token in sent if not token.is_punct)for sent in sentences)
        sentence_lengths.extend([len(sen) for sen in sentences])
        try:
            sentence_length_difference = max(sentence_lengths) - min(sentence_lengths)
        except ValueError:
            sentence_length_difference = 0
        sentence_lengths_words = [len([token for token in sent if not token.is_punct])
                            for sent in sentences]
        std_words_per_sentence = np.std(sentence_lengths_words)
        sentence_length_difference_list.append(sentence_length_difference)
        total_tokens += len(doc)
        words_alpha = [t for t in doc if t.is_alpha]
        len_of_words = [len(t.text) for t in doc if t.is_alpha]
        length_of_words_list.extend(len_of_words)
        longest_word = retrieve_maximum_safely(len_of_words)
        shortest_word = retrieve_minimum_safely(len_of_words)
        total_chars = sum(len(t.text) for t in words_alpha)

        # lexical features
        flesch_reading_easiness, flesch_grade, dale_chall, ari, gf = lf.get_reading_easiness(doc.text) # normal text
        spelling_errors = lf.spelling_errors(doc.text) # one document

        # reading easiness metrics
        flesch_reading_easiness_sum += flesch_reading_easiness
        flesch_grade_sum += flesch_grade
        dale_chall_sum += dale_chall
        ari_sum += ari
        gf_sum += gf

        spelling_errors_sum += spelling_errors


        difference_between_longest_and_shortest_word.append(longest_word-shortest_word)
        #chars_per_doc.append(save_division(total_chars, len(words_alpha)))

        # dictionary features
        # get the mean from there

        (words_with_two_or_three_chars,
         count_stopwords, count_acronyms,
         count_first_person_pronouns) = lf.wrapper_for_spacy(doc)

        count_stopwords_sum += count_stopwords
        count_acronyms_sum += count_acronyms
        count_first_pronoun_sum += count_first_person_pronouns


        words_with_len_two += words_with_two_or_three_chars[0]
        words_with_len_three += words_with_two_or_three_chars[1]



        try:
            dis_analysis_results = discf.run_analysis(text)     # discours
        except IndexError:
            text=""
            dis_analysis_results = discf.run_analysis(text)
        imbalance_root = dis_analysis_results["imbalance_root"]
        total_imbalance = dis_analysis_results["total_imbalance"]

        edu_freq = dis_analysis_results["edu_freq"]
        edu_freq_count = dis_analysis_results["edu_freq_count"]

        edu_freq_nuc = dis_analysis_results["edu_freq_nuc"]
        edu_freq_nuc_count = dis_analysis_results["total_relations_edu_and_satellite"]

        nuclearity_freq = dis_analysis_results["nuclearity_freq"]
        nuclearity_freq_count = dis_analysis_results["total_nuclearity_counts"]

        edu_freq_count_total += edu_freq_count
        edu_freq_nuc_total += edu_freq_nuc_count
        nuclearity_freq_count_total += nuclearity_freq_count

        # Calculate the mean later
        imbalance_root_list.append(imbalance_root)
        total_imbalance_list.append(total_imbalance)

        edu_freq_sum = add_values_to_dict(edu_freq_sum, edu_freq) # update the counter with the dictionaries
        edu_freq_nuc_sum = add_values_to_dict(edu_freq_nuc_sum, edu_freq_nuc)
        nuclearity_freq_sum = add_values_to_dict(nuclearity_freq_sum, nuclearity_freq)



        extracted_syntactic_features = sf.wrap_syntactic_features(doc)    # char feature

        total_past += extracted_syntactic_features["past_count"]
        total_present += extracted_syntactic_features["present_count"]
        total_verbs += extracted_syntactic_features["verb_count"]
        ner_count += extracted_syntactic_features["ner_count"]

        total_tokens_sum += extracted_syntactic_features["total_tokens"]
        freq_of_gram_cats_sum = add_values_to_dict(freq_of_gram_cats_sum, extracted_syntactic_features["freq_of_gram_cats"])
        freq_gram_freqs_common_sum = add_values_to_dict(freq_gram_freqs_common_sum, extracted_syntactic_features["gram_freqs_common"])
        dependendency_relations_per_sentence_sum = add_values_to_dict(dependendency_relations_per_sentence_sum,
                                                                      extracted_syntactic_features["dependency_relations_per_sentence"])






        ramification_factor_list.append(extracted_syntactic_features["ramification_factor"])
        tree_width_list.append(extracted_syntactic_features["tree_width"])
        tree_depth_list.append(extracted_syntactic_features["tree_depth"])

        #dependendency_relations_per_sentence_sum += dependendency_relations_per_sentence
        freq_modifiers_sum += extracted_syntactic_features["count_modifiers"]
        total_relations_sum += extracted_syntactic_features["total_relations"]
        freq_adv_dep_sum += extracted_syntactic_features["freq_adv_dep"]
        freq_mod_verbs_sum += extracted_syntactic_features["mod_verbs"]
        freq_verb_complex_tenses_sum += extracted_syntactic_features["freq_verb_complex_tenses"]
        freq_comparative_and_superlative_adjectives_and_adverbs_sum += extracted_syntactic_features["freq_comparative_superlative_adj_adv"]


    # mean chars per word
    mean_chars_per_word = save_division(total_chars, total_words)

    # calculate means and ratios
    mean_no_of_words_per_sent = save_division(total_words, sentence_count)
    percentage_of_modifiers = save_division(freq_modifiers_sum, total_relations_sum)
    freq_adv_dep_avg = save_division(freq_adv_dep_sum, sentence_count)
    freq_mod_verbs_avg = save_division(freq_mod_verbs_sum, total_verbs)
    freq_verb_in_complex_tenses = save_division(freq_verb_complex_tenses_sum, total_verbs)

    present_avg = save_division(total_present, total_verbs)
    past_avg = save_division(total_past, total_verbs)
    ner_avg = save_division(ner_count, total_tokens)

    freq_comparative_and_superlative_adjectives_and_adverbs_avg = save_division(
        freq_comparative_and_superlative_adjectives_and_adverbs_sum, total_tokens)

    words_with_len_two_ratio = save_division(words_with_len_two, total_words)
    words_with_len_three_ratio = save_division(words_with_len_three, total_words)

    stopwords_ratio = save_division(count_stopwords_sum, total_words)
    acronyms_ratio = save_division(count_acronyms_sum, total_words)
    fpo_ratio = save_division(count_first_pronoun_sum, total_words)

    mean_ramification_factor = np.mean(ramification_factor_list)
    mean_tree_width = np.mean(tree_width_list)
    mean_tree_depth = np.nanmean(tree_depth_list) # use namean because list can contain nan due to recursion error (happned 1x for reddit)
    # added 18.06sd
    std_ramification_factor = np.std(ramification_factor_list)
    std_tree_width = np.std(tree_width_list)
    std_tree_depth = save_std(tree_depth_list)

    std_deviation_of_word_length = np.std(length_of_words_list)

    # discourse
    mean_imbalance_root = np.mean(imbalance_root_list)
    mean_total_imbalance = np.mean(total_imbalance_list)

    std_imbalance_root = np.std(imbalance_root_list)
    std_total_imbalance = np.std(total_imbalance_list)

    sentence_length_difference_mean = np.mean(sentence_length_difference_list)




    #average the dictionaries
    freq_of_gram_cats_avg = average_dictionary_vals_by_count(freq_of_gram_cats_sum, total_tokens_sum) #average_dictionary_vals_mean(freq_of_gram_cats_sum)
    freq_gram_freqs_common_avg = average_dictionary_vals_by_count(freq_gram_freqs_common_sum, total_tokens_sum)

    # take all dependency relations and divide them by the amount of sentences
    dependendency_relations_per_sentence_avg = average_dictionary_vals_mean(dependendency_relations_per_sentence_sum)
    edu_freq_avg = average_dictionary_vals_by_count(edu_freq_sum,edu_freq_count_total)
    edu_freq_nuc_avg = average_dictionary_vals_by_count(edu_freq_nuc_sum, edu_freq_nuc_total)
    nuclearity_freq_avg = average_dictionary_vals_by_count(nuclearity_freq_sum, nuclearity_freq_count_total)



    features = {
        "flesch_kincaid_easiness": flesch_reading_easiness_sum / count_of_documents,
        "flesch_grade": flesch_grade_sum / count_of_documents,
        "dale_chall": dale_chall_sum / count_of_documents,
        "ari": ari_sum / count_of_documents,
        "gf": gf_sum / count_of_documents,
        "discourse_markers_ratio": disc, # all text
        "abbreviations_ratio": abb, # all text
        "positive_ratio": pos, # all text
        "negative_ratio": neg, # all text
        "profanity_ratio": prof, # all text""
        "interjections_ratio": interjections,
        "total_numbers_of_chars": total_number_of_chars, # calulate the average the amount of documents
        "avg_chars_per_document": total_number_of_chars/count_of_documents,
        "upper_ratio": upper_ratio, # done
        "lower_ratio": lower_ratio, # done
        "ratio_of_periods": ratio_of_periods, # done
        "ratio_of_commas": ratio_of_commas, # done
        "ratio_of_semicolons": ratio_of_semicolons, # done
        "ratio_of_exclamation": ratio_of_exclamation, # done
        "ratio_of_parentheses": ratio_of_parentheses, # done
        "ratio_of_numbers": ratio_of_numbers, # done
        "ratio_of_hyphens": ratio_of_hyphens, # done
        "ratio_of_colons":ratio_of_colons, # done
        "ratio_of_quotation_marks": ratio_of_quotation_marks, # done        "spelling_errors": spelling_errors_sum / count_of_documents
        "spelling_errors": spelling_errors_sum / count_of_documents,
        "mean_n_of_words_per_sent": mean_no_of_words_per_sent,  # total words divded by sentence count"
        "std_words_per_sentence": std_words_per_sentence, #
        "diff_between_min_and_max_sent": sentence_length_difference_mean,  # done
        "average_word_length":avg_w_length, # done, on all texts
        "maximum_word_length":maximum_w_length, # done, on all texts
        "minimum_word_length":minimum_w_length,  # done, on all texts
        "std_w_length": std_w_length,
        "ttr":ttr, # done, on all texts
        "cttr":cttr, # done, on all texts
        "mean_number_of_chars_per_word": mean_chars_per_word,  # done
        "words_with_two_chars_ratio": words_with_len_two_ratio,#done
        "words_with_three_chars_ratio": words_with_len_three_ratio, #done
        "standard_deviation_of_word_length": std_deviation_of_word_length,
        "diff_longest_and_shortest_word": max(difference_between_longest_and_shortest_word), #done
        "ratio_stopwords": stopwords_ratio, # done
        "ratio_acronyms": acronyms_ratio, # done
        "ratio_first_person_pronouns": fpo_ratio, # done
        "mean_ramification_factor":mean_ramification_factor, # done
        "mean_tree_width": mean_tree_width, # done
        "mean_tree_depth": mean_tree_depth, # done
        "std_ramification_factor": std_ramification_factor,
        "std_tree_width": std_tree_width,
        "std_tree_depth": std_tree_depth,
        "percentage_of_modifiers": percentage_of_modifiers, # done
        "freq_adv_dep": freq_adv_dep_avg, # freq of adverbial dependencies
        "ratio_of_freq_mod_verbs": freq_mod_verbs_avg, # frequency of modal verbs
        "freq_comparative_and_superlative_adjectives_and_adverbs":
            freq_comparative_and_superlative_adjectives_and_adverbs_avg,
        "percentage_verbs_in_complex_tenses": freq_verb_in_complex_tenses,
        "present_ratio": present_avg, #done
        "past_ratio": past_avg,  #done
        "ner_ratio": ner_avg,  #done
        "mean_imbalance_root": mean_imbalance_root,
        "mean_total_imbalance": mean_total_imbalance,
        "std_imbalance_root": std_imbalance_root,
        "std_total_imbalance": std_total_imbalance
    }
    features.update(freq_of_gram_cats_avg)
    features.update(freq_gram_freqs_common_avg) # merged grammatical categories
    features.update(dependendency_relations_per_sentence_avg)
    features.update(edu_freq_avg)
    features.update(edu_freq_nuc_avg)
    features.update(nuclearity_freq_avg)
    #features.update(dependendency_relations_per_sentence)
    features_final = {str(k): v for k, v in features.items()} # convert all keys to strings to precent index error
    print("Features")
    print(features_final)
    return pd.Series(features_final)

if __name__ == "__main__":
    n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    #input_file = "/home/philipp.meier/author_profiling_generalizability/data/final/pan_2014_reviews_full_final.tsv" #sys.argv[1] #pan_2017_Twitter_male_ap.tsv
    input_file = sys.argv[1]
    output_file_ = input_file.split(".tsv")[0]
    output_file = output_file_ + "_full_features_4.tsv"
    data = pd.read_csv(input_file,
                          delimiter='\t',quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,
                            lineterminator="\n")
    #data = data.head()


    print(data.columns)
    data["document"] = data["document"].apply(ast.literal_eval)
    #data["document"] = data["document"].fillna("").astype(str)
    print(data["document"])

    data = data.fillna('')
    features = data["document"].apply(feature_extraction)
    print(features)
    df_new = pd.concat([data["author_id"], features, data["label"]], axis=1)
    print(df_new)
    df_new.to_csv(output_file, sep="\t")
