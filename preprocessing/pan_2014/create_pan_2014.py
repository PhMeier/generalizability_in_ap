"""
@date: 04.03.2026
"""
import os
import csv
import sys
from lxml import etree

def pan_data_2014_mapping(path, f):
    """
    Count the total amount of entries in a xml file for each gender.
    For PAN 2014, 2015
    :param f:
    :param c:
    :return:
    """
    filename = path + f
    c = 0
    with open(filename, "r", encoding="utf-8") as f: # UnicodeDecodeError: 'utf-8' codec can't decode byte 0x92 in position 1131: invalid start byte
        for line in f:
            line = line.strip()
            if "<document id=" in line:
                c+=1
    return c



def read_xml_file(filename):
    data = []
    parser = etree.XMLParser(recover=True, encoding="utf-8")
    tree = etree.parse(filename, parser)
    root = tree.getroot()
    for post in root.iter("document"):
        if post.text:
            text = post.text.strip()
            text = text.replace("\n", "")
            data.append(text)
    return data

def sanitize_text(text):
    """
    Reove non printable asci characters (\x00), otherwise, pandas reads such lines as "nan" and we loose information.
    :param text:
    :return:
    """
    if not text:
        return ""
    return "".join(c for c in text if c.isprintable() or c in "\n\t\r")


def write_to_csv(data_dict: dict, outputfile:str):
    """
    all_male_blog_posts.tsv
    all_female_blog_posts.tsv
    Quote all fields, since blog posts are prone to be messy
    Escapechar \\to handle quotes.
    Use \n as linterminator for compatibility on Linux and Windows

    :param data: list of lists
    :param outputfile:
    :return:
    """
    with open(outputfile, "w+", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter='\t',quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,
                            lineterminator="\n")
        header = ["author_id", "document"]
        writer.writerow(header)
        for key, all_blogposts in data_dict.items():
            #post = " ".join(all())
            for post in all_blogposts:
                post = sanitize_text(post)
                writer.writerow([key, post])


def create_csv_from_files(filename_to_gender, path, gender, outputfile):
    d = {}
    for key, val in filename_to_gender.items():
        d[key] = []
        if val == gender: # 2014 MALE or FEMALE
            # open file
            content = read_xml_file(path+key)
            d[key].extend(content)
    write_to_csv(d, outputfile)





if __name__ == "__main__":
    gender = sys.argv[1] # either male or female
    assert gender == "MALE" or gender == "FEMALE"

    truth_rev = "/pan_ap14/en_reviews/pan14-author-profiling-training-corpus-english-reviews-2014-04-16/truth.txt"
    path = "/pan_ap14/en_reviews/pan14-author-profiling-training-corpus-english-reviews-2014-04-16/"
    outputfile = f"pan_2014_reviews_{gender}.tsv"

    truth_file = truth_rev

    filename_to_gender = {}
    gender_to_file_name_counts = {}
    gender_to_file_name_counts["MALE"] = 0
    gender_to_file_name_counts["FEMALE"] = 0
    with open(truth_file, "r", encoding="utf-8") as f:
        for line in f:
            filename = line.split(":::")[0].strip() + ".xml"
            gender = line.split(":::")[1].strip()
            filename_to_gender[filename] = gender
            gender_to_file_name_counts[gender] = 1 + gender_to_file_name_counts.get(gender, 0)
    print(filename_to_gender)
    print(gender_to_file_name_counts)
    c = 0
    create_csv_from_files(filename_to_gender, path, gender, outputfile)

