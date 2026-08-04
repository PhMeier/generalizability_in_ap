"""
@date: 04.03.2026
"""
import os
import sys
import csv
from lxml import etree



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


def create_csv_from_files(filename_to_gender, outputfile):
    pan_2017_tweets_test = "/pan_ap_2017/ap_test_17/pan_ap_17_test/en/"

    d = {}
    for key, val in filename_to_gender.items():
        d[key] = []
        if val == "male": # 2014, 2016: MALE
            # open file
            content = read_xml_file(pan_2017_tweets_test+key)
            d[key].extend(content)
    write_to_csv(d, outputfile)



if __name__ == "__main__":
    """
    Creates a tsv file either for males or females for the training set of PAN 2017
    """

    gender = sys.argv[1] # either male or female
    assert gender == "male" or gender == "female"
    #pan_2017_tweets = "/pan17-author-profiling-training-dataset-2017-03-10/en/"
    pan_2017_train_truth = "/pan17-author-profiling-training-dataset-2017-03-10/en/truth.txt"


    outputfile = f"pan_2017_train_twitter_{gender}.tsv"

    truth_file = pan_2017_train_truth

    filename_to_gender = {}
    gender_to_file_name = {}
    gender_to_file_name["MALE"] = 0
    gender_to_file_name["FEMALE"] = 0
    with open(truth_file, "r", encoding="utf-8") as f:
        for line in f:
            filename = line.split(":::")[0].strip() + ".xml"
            gender = line.split(":::")[1].strip()
            filename_to_gender[filename] = gender
            gender_to_file_name[gender] = 1+gender_to_file_name.get(gender, 0)
    print(filename_to_gender)
    print(gender_to_file_name)
    c = 0
    create_csv_from_files(filename_to_gender, outputfile)




