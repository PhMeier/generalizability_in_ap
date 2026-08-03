import os
import csv
from enum import Enum
from lxml import etree
from os import listdir
from os.path import isfile, join


GENDER_INDEX_PAN_13 = 1
AGE_INDEX_PAN_13 = 2
PEDOPHILE_INDEX = -1
SEPARATOR = ":::"
class Gender_Pan_13(Enum):
    M="male"
    F="female"

"""
Reads in the PAN 2013 data (Training and Test).
Filters chats from sexual predators. 
Creates a tsv where each row contains one document of an author. This is the input for html_parsing
"""


def sanitize_text(text):
    """
    Reove non printable asci characters (\x00), otherwise, pandas reads such lines as "nan" and we loose information.
    :param text:
    :return:
    """
    if not text:
        return ""
    return "".join(c for c in text if c.isprintable() or c in "\n\t\r")


def read_xml(filename, keyword="conversation"):
    data = []
    parser = etree.XMLParser(recover=True, encoding="utf-8")
    tree = etree.parse(filename, parser)
    root = tree.getroot()
    for post in root.iter(keyword): #("document"):
        if post.text:
            text = post.text.strip()
            text = text.replace("\n", "")
            text = sanitize_text(text)
            data.append(text)
    return " ".join(data), data # concatenate the text to get an author representation A

def write_to_csv(outputfile, data, label):
    if "female" in outputfile and label != 1:
        return "False Label! Female authors receive label Zero, Not One"
    with open(outputfile, "w", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f, delimiter='\t',quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,
                            lineterminator="\n")
        csv_writer.writerow(["author_id", "text", "label"])
        for key, value in data.items():
            csv_writer.writerow([key, value, label])

def write_to_csv_sg(outputfile, data, label):
    if "female" in outputfile and label != 1:
        return "False Label! Female authors receive label Zero, Not One"
    with open(outputfile, "w+", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f, delimiter='\t',quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,
                            lineterminator="\n")
        csv_writer.writerow(["author_id", "text", "label"])
        for key, value in data.items():
            for instance in value:
                csv_writer.writerow([key, instance, label])

def retrieve_all_xml_files(path, truth_file, partition):
    male_content_ap = {}
    female_content_ap = {}
    male_content_sg = {} # contains the single instances, one line per utterancce
    female_content_sg = {}
    with open(truth_file, "r", encoding="utf-8") as f:
        for line in f:
            if line != "\n":
                if line.split(SEPARATOR)[-1].strip() not in ["pedophile", "sex"]: # fitler out sexual predators
                    age = line.split(SEPARATOR)[AGE_INDEX_PAN_13].strip()
                    gender = line.split(SEPARATOR)[GENDER_INDEX_PAN_13].strip()
                    filename = line.split(SEPARATOR)[0].strip() + "_en_XXX_XXX.xml"
                    if gender == Gender_Pan_13.M.value:
                        content, data = read_xml(path+filename)
                        male_content_ap[filename] = content
                        male_content_sg[filename] = data
                    if gender == Gender_Pan_13.F.value:
                        content, data = read_xml(path+filename)
                        female_content_ap[filename] = content
                        female_content_sg[filename] = data
                else:
                    print(line)
    #write_to_csv(f"pan_13_test_male.tsv", male_content_ap, 0)
    #write_to_csv(f"pan_13_test_female.tsv", female_content_ap, 1)

    # for each document, write one row
    write_to_csv_sg(f"pan_13_{partition}_male.tsv", male_content_sg, 0)
    write_to_csv_sg(f"pan_13_{partition}_female.tsv", female_content_sg, 1)

def pan_2013_test():
    """
    PAN 2013 Test routing
    :return:
    """
    partition = "test"
    path = "./PAN AP 2013/pan13-author-profiling-test-and-training/pan13-author-profiling-test-corpus2-2013-04-29/pan_13_test/en/"
    truth_file = "./PAN AP 2013/pan13-author-profiling-test-and-training/pan13-author-profiling-test-corpus2-2013-04-29/pan_13_test/truth-en.txt"
    retrieve_all_xml_files(path, truth_file, partition)

def pan_2013_train():
    partition = "training"
    path = "./PAN AP 2013/pan13-author-profiling-test-and-training/pan13-author-profiling-training-corpus-2013-01-09/pan13-author-profiling-training-corpus-2013-01-09/en/"
    truth_file = "./PAN AP 2013/pan13-author-profiling-test-and-training/pan13-author-profiling-training-corpus-2013-01-09/pan13-author-profiling-training-corpus-2013-01-09/truth-en.txt"
    retrieve_all_xml_files(path, truth_file, partition)

if __name__ == "__main__":
    ...
    #pan_2013_test()
    #pan_2013_train()






