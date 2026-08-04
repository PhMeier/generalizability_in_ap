# PAN 2013

Get the PAN 2013 dataset from [here.](https://zenodo.org/records/3715864)

1) Read in the dataset and create training and test partition. Creates pan_13_test_male.tsv, pan_13_test_female.tsv
pan_13_train_male.tsv and pan_13_train_female.tsv
```
python3 read_in_pan13.py
```
2) Run html_parsing.py to parse the html data and clean the data. Creates pan_13_test_preprocessed_URL_masked.tsv and
pan_13_train_preprocessed_URL_masked.tsv
```
python3 html_parsing.py
```
3) Downsample the data (Only training partition)
```
python3 downsample_data.py
```
4) Create profiles for training, val and test for a specific classifier.
```
python3 create_profiles.py bert
```