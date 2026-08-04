"""
Preprocessing blog corpus which comes as html files.
Name of the xml files contain the author attributes, (e.g 5114.male.25.indUnk.Scorpio)
The content of the html files is structured as follows:
<Blog>
<date>01.01.2000</date>
<post>The internet is still working in the New Year!</post>
<date>02.01.2000</date>
<post>The internet is still working!</post>
<Blog>
"""
import csv
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from lxml import etree
#import matplotlib.plt as plt
import re


def read_file_content_greedy(filename):
    """
    The blog authorship corpus contains empty posts which are discarded when using .strip() and html formatting.
    This function reproduces the original reported number of blog posts
    Orig numbers:
    681288 Total
    345197 for male
    336091 for female
    336091 for female raw
    345197 for male raw
    334701 for female cleaned (only excluded empty posts)
    343502 for male cleaned (only excluded empty posts)
    :param filename:
    :return:
    """
    with open(filename, encoding="ISO-8859-1") as f:
        content = f.read()

    # Extract everything between <post> and </post>, including newlines
    posts = re.findall(r"<post>(.*?)</post>", content, re.DOTALL | re.IGNORECASE)

    cleaned = [post.strip() for post in posts if post.strip()]

    return cleaned

URL_EXTRACT_PATTERN_14 = re.compile(r'https?://(?:[/\-\w.]|(?:%[\da-fA-F]{2}))+')

def sanitize_text(text):
    """
    Reove non printable asci characters (\x00), otherwise, pandas reads such lines as "nan" and we loose information.
    :param text:
    :return:
    """
    if not text:
        return ""
    text = "".join(c for c in text if c.isprintable() or c in "\n\t\r")
    text = re.sub(URL_EXTRACT_PATTERN_14, "[URL]", text)
    text = text.replace("urlLink", "[URL]")
    return "".join(c for c in text if c.isprintable() or c in "\n\t\r")


def generate_csv_per_gender(data_dict: dict, label, outputfile:str):
    """
    all_male_blog_posts.tsv
    all_female_blog_posts.tsv
    Quote all fields, since blog posts are prone to be messy
    Escapechar \\to handle quotes.
    Use \n as linterminator for compatibility on Linux and Windows
    Encode the file as "ISO-8859-1" to avoid Decoding Errors.

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
                writer.writerow([key, post, label])


def read_in_file(files:list):
    """
    Read in the files from a directory and create a dictionary where the key is the filename (author) and the value
    is the file content
    :param files:
    :return d:
    """
    d = {}
    for f in files:
        data = []
        author_id = f.split("/")[-1]
        data.extend(read_file_content_greedy(f))
        if author_id in d:
            d[author_id].extend(data)
        else:
            d[author_id] = data
    return d

def generate_csv_procedure():
    """
    For every line, there is one blog post by one author. Needs to ge grouped in order to create author representations
    :return:
    """
    directory = "../data/blogs/*" # pascal

    files = glob.glob(directory)

    male = [f for f in files if ".male." in f]
    female = [f for f in files if ".female." in f]

    print(len(male))
    print(len(female))
    female_data = read_in_file(female)
    male_data = read_in_file(male)
    flattened_list_females = [x for xs in female_data.values() for x in xs]
    flattened_list_males = [x for xs in male_data.values() for x in xs]
    print(len(flattened_list_females))
    print(len(flattened_list_males))
    generate_csv_per_gender(male_data, 0, "all_male_blog_posts_raw.tsv")
    generate_csv_per_gender(female_data, 1, "all_female_blog_posts_raw.tsv")


if __name__ == "__main__":
    directory = "../../data/blogs/*"

    # check for .male. and .female.
    files = glob.glob(directory)
    #print(files)
    male = [f for f in files if ".male." in f]
    female = [f for f in files if ".female." in f]

    print(len(male))
    print(len(female))
    female_data = read_in_file(female)
    male_data = read_in_file(male)
    flattened_list_females = [x for xs in female_data.values() for x in xs]
    flattened_list_males = [x for xs in male_data.values() for x in xs]
    print(len(flattened_list_females))
    print(len(flattened_list_males))

    generate_csv_per_gender(male_data, 0,"all_male_blog_posts_cleaned.tsv")
    generate_csv_per_gender(female_data, 1, "all_female_blog_posts_cleaned.tsv")

