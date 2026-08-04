# Pan 2017

Get the PAN 2017 dataset from [here.](https://zenodo.org/records/3745980)

1) Run create_pan_2017_train.py and retrieve_pan_2017_test.py. This creates a train and test file for each gender.
```
python3 create_pan_2017_train.py
python3 retrieve_pan_2017_test.py
```

2) Run clean_data.py. This merges male and female data and cleans URLs. Creates a merged dataset.
```
python3 clean_data.py pan_2017_train_twitter_male.tsv pan_2017_train_twitter_female.tsv train
python3 clean_data.py pan_2017_test_twitter_male.tsv pan_2017_test_twitter_female.tsv test
```
3) Run create_profiles.py. Creates the author representation U for the input file and includes a separator token.
```
python3 create_profiles.py pan_2017_train_prepro.tsv bert
python3 create_profiles.py pan_2017_test_prepro.tsv bert
```
4) Run split_data.py (For training data only!). Required to split the training data in 80% training and 20% validation.
Split is stratified. 
```
python3 split_data.py --input pan_2017_train_prepro.tsv --data pan_17
```