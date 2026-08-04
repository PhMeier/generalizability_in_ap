# Generalizability in Author Profiling

This repository creates the code for the experiments in "Generalizability in Author Profiling".

- Preprocessing contains the code for preprocessing each dataset. The source of each dataset is linked in the ReadMe file
of the directory.
  - Each URL is masked with [URL]
  - If the data is masked, the data is masked first since stanza handles separator tokens incorrectly
  - At last, the representation U is created for each author. For the language models, texts are concatenated by using
  the models separator token between each document. This does not apply to datasets for LLMs and white-box models.
- Feature extraction contains the code for feature extraction for white box models
- Finetuning contains code to fine-tune the black-box models.
- Hyperparameter search contains the code for hyperparameter search for black box and white box models
- Inference contains code for creating the results for black-box and white-box models as well as the LLM.

Code requires Python 3.10 or higher. Requirements are listed in requirements.txt and can be used to create a conda 
environment