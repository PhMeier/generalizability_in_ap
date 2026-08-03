import os
import csv
from enum import Enum
from lxml import etree
from os import listdir
from os.path import isfile, join


GENDER_INDEX_PAN_17 = 1


class PAN_2017(Enum):
    M="male"
    F="female"

def read_xml(filename, keyword="document"):
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

def write_to_csv_ap(outputfile, data, label):
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


def sanitize_text(text):
    """
    Remove non printable asci characters (\x00), otherwise, pandas reads such lines as "nan" and we loose information.
    :param text:
    :return:
    """
    if not text:
        return ""
    return "".join(c for c in text if c.isprintable() or c in "\n\t\r")


def retrieve_all_xml_files(path, truth_file, separator):
    male_content_ap = {}
    female_content_ap = {}
    male_content_sg = {} # contains the single instances, one line per utterancce
    female_content_sg = {}
    with open(truth_file, "r", encoding="utf-8") as f:
        for line in f:
            if line != "\n":
                filename = line.split(separator)[0].strip() + ".xml"
                gender = line.split(separator)[GENDER_INDEX_PAN_17].strip()
                if gender == PAN_2017.M.value:
                    content, data = read_xml(path+filename)
                    male_content_ap[filename] = content
                    male_content_sg[filename] = data
                if gender == PAN_2017.F.value:
                    content, data = read_xml(path+filename)
                    female_content_ap[filename] = content
                    female_content_sg[filename] = data
    #write_to_csv_ap("pan_17_test_male_ap.tsv", male_content_ap, 0)
    #write_to_csv_ap("pan_17_test_female_ap.tsv", female_content_ap, 1)

    write_to_csv_sg("pan_17_test_male_sg.tsv", male_content_sg, 0)
    write_to_csv_sg("pan_17_test_female_sg.tsv", female_content_sg, 1)



if __name__ == "__main__":
    path = "/pan_17_test/pan17-author-profiling-test-dataset-2017-03-16/en/"
    truth_file = path+"truth.txt"
    separator = ":::"
    retrieve_all_xml_files(path, truth_file, separator)