# PAN 2014

Get the PAN 2014 dataset from [here.](https://zenodo.org/records/3716008) and use the Hotel reviews.

1) Read in the PAN 2014 data and create a tsv for each gender.
```
python3 create_pan_2014_train.py
```
2) Clean data. Creates pan_2014_reviews_prepro.tsv
```
python3 clean_data.py pan_2014_reviews_MALE.tsv
python3 clean_data.py pan_2014_reviews_FEMALE.tsv
```
3) Create Author profiles for every classifier
```
python3 create_profiles.py pan_2014_reviews_prepro.tsv bert
```
4) Split the data intro training, dev and test.
```
python3 split_data.py --input pan_2014_reviews_prepro_bert_ap.tsv --data pan14
```