# Reddit

Dataset is available here: https://psy.takelab.fer.hr/datasets/all/mbti9k/#terms-of-use

The original test partition for Reddit is under ./pandora_baseline/res/is_female-LR-N.
For the training data, all_comments_since_2015 and author_profiles.csv was merged

1) First run "merge_gender_information.py". This creates for male and female authors a file, where each row contains
 one author, the concatenation U of the documents and the document count. 
2) For the test set, run create_test_author_info_to_comment.py
2) This data is used to downsample the training, validation and test partition.
3) Run create_posts_per_row.py afterwards to create a file which contains one document per row
4) Run create_prfoles to create the input for black box models and white box models.