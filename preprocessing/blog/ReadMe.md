# Blog

Get the Blog Authorship corpus from [here.](https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm)

1) First run clean_blog.py. Needs the path blog: ./../data/blogs/*
```
python3 clean_blog.py
```
2) Downsample the blog data, which splits the data into training, dev and test. Takes in all_male_blog_posts_cleaned.tsv 
and all_female_blog_posts_cleaned.tsv
```
python3 downsample_blog.py
```
3) Create Author Profiles for each classifier and the feature pipeline with with create_author_profiles. create_author_profiles.py takes in
   the downsampled data of downsample_blog.py as input: blog_sampled_30000000_train.tsv, blog_sampled_10000000_val.tsv, 
   blog_sampled_10000000_test.tsv
```
python3 create_profiles.py bert
```