# Reddit

Dataset is available here upon request: https://psy.takelab.fer.hr/datasets/all/mbti9k/#terms-of-use
Due to licence, we can not share the data.
The original test partition for Reddit is under ./pandora_baseline/res/is_female-LR-N.
For the training data, all_comments_since_2015 and author_profiles.csv was merged

1) First run "merge_gender_information.py". This creates for male and female authors a file, where each row contains
 one author, the concatenation U of the documents and the document count. Creates reddit_train_male_ap.tsv and 
reddit_train_female_ap.tsv
```
python3 merge_gender_information.py
```
2) For the test set, run create_test_author_info_to_comment.py. Creates reddit_test/pandora_author_info_to_comment.tsv
```
python3 create_test_author_info_to_comment.py
```
3) This data is used to downsample the training, validation and test partition.
```
python3 downsample_reddit.py
```
4) Run create_posts_per_row.py afterwards to create a file which contains one document per row. Takes in
reddit_sampled_no_sort_2000000_test.tsv, reddit_sampled_no_sort_30000000_train.tsv, reddit_sampled_no_sort_2000000_val.tsv
Creates reddit_sampled_no_sort_30000000_train_doc_per_line.tsv, reddit_sampled_no_sort_2000000_val_doc_per_line.tsv, 
reddit_sampled_no_sort_2000000_test_doc_per_line.tsv
```
python3 create_posts_per_row.py
```
5) Run create_profiles to create the input for black box models and white box models.
```
python3 create_profiles.py bert
```