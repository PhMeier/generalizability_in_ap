import json
import pandas as pd


def read_txt_file(f):
    res = []
    with open(f, "r", encoding="utf-8") as f:
        for line in f:
            res.append(line.lower().strip())
    return res


class DictionaryFeatures():
    def __init__(self):
        f_inj = "interjections.txt"
        self.inj = read_txt_file(f_inj)
        f_discourse_markers = "discourse_markers.txt"
        self.disc = read_txt_file(f_discourse_markers)
        # Abbreviations list
        # https://github.com/ipekdk/abbreviation-list-english/blob/main/abbreviations_eng.xls
        f_abb = "abbreviations.txt"
        self.abb = read_txt_file(f_abb)
        # positive words
        # https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
        f_pos = "positive-words.txt"
        self.pos = read_txt_file(f_pos)
        # negative words
        f_neg = "negative-words.txt"
        self.neg = read_txt_file(f_neg)
        # profane words
        f_prof = "words.json"
        with open("words.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.profanity_words = [entry["word"] for entry in data]
        #print(words)

    def count_interjections_markers(self, row):
        if row:
            c=0
            row_words = row.lower().split()
            for word in self.inj:
                if word in row_words:
                    c+=1
            return c/len(row_words)
        else:
            return 0

    def count_discourse_markers(self, row):
        if row:
            c=0
            row_words = row.lower().split()
            for word in self.disc:
                if word in row_words:
                    c+=1
            return c/len(row_words)
        else:
            return 0

    def count_abb(self, row) -> float:
        if row:
            c=0
            row_words = row.lower().split()
            for word in self.abb:
                if word in row_words:
                    c+=1
            return c/len(row_words)
        else:
            return 0
    def count_pos(self, row) -> float:
        if row:
            c=0
            row_words = row.lower().split()
            for word in self.pos:
                if word in row_words:
                    c+=1
            return c/len(row_words)
        else:
            return 0

    def count_neg(self, row) -> float:
        if row:
            c=0
            row_words = row.lower().split()
            for word in self.neg:
                if word in row_words:
                    c+=1
            return c/len(row_words)
        else:
            return 0


    def count_profanity(self, row) -> float:
        if row:
            c=0
            row_words = row.lower().split()
            for word in self.profanity_words:
                if word in row_words:
                    c+=1
            return c/len(row_words)
        else:
            return 0

