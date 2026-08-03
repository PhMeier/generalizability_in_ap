# Masking

There are two files to mask the data. One for BERT and the other file for RoBERTa and XLNET.
Both files require the data to have masked URLs and no separator tokens between the documents!
Separator tokens are not correctly handled by stanza and are inserted after the masking. 

1) Run the masking methods
2) Add the separator tokens in the postprocess.