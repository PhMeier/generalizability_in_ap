# Masking

There are two files to mask the data. One for BERT and the other file for RoBERTa and XLNET.
Both files require the data to have masked URLs and no separator tokens between the documents!
Separator tokens are not correctly handled by stanza and are inserted after the masking. 

1) Run the masking methods. File needs to be in representation format U for baselines (texts saved as lists)
```
python3 mask_data_for_bert.py
python mask_data_roberta_and_xlnet.py
```
2) Add the separator tokens in the postprocess.
```
python3 postprocess_add_separator_token_for_masked_data.py --input input_file_BERT_masked.tsv --model bert

```