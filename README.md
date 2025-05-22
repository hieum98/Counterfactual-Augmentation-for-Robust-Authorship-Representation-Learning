# *Counterfactual Augmentation for Robust Authorship Representation Learning*

[![SIGIR](https://img.shields.io/badge/SIGIR-2024-b31b1b.svg)](https://dl.acm.org/doi/pdf/10.1145/3626772.3657956)

ERLAS is official pytorch implementation for the paper "Counterfactual Augmentation for Robust Authorship Representation Learning". In this framework we introduce generating style-counterfactual examples by retrieving the most similar content texts by different authors on the same topics/domains.

## Installation
To use GenC, install evironment from ```requirements.txt```
```bash
pip install -r requirements.txt
```

After that, you can install our package from source by
```bash
pip install -e .
```

## Data Processing

You should download following datasets manually and processing them into format as in `sample_data/samples.jsonl`:

### Reddit

Execute the following command to download the Reddit data:

```bash
./scripts/download_reddit_data.sh
```

### Amazon

The amazon data must be requested from [here](https://nijianmo.github.io/amazon/index.html#files) (the "raw review data" (34gb) dataset). 

### Fanfiction

The fanfiction data must be requested from [here](https://zenodo.org/record/3724096#.YT942y1h1pQ)


## Traning

To train the model on Amazon dataset, run the following command:
```bash
python ERLAS/main.py \
    --dataset_name Amazon \
    --token_max_length 128 \
    --index_by_BM25 \
    --BM25_percentage 1.0 \
    --gpus 2 \
    --use_gc \
    --gc_minibatch_size 16 \
    --learning_rate 2e-5 \
    --learning_rate_scaling \
    --num_epoch 3 \
    --do_learn \
    --experiment_id Amazon \
    --version unseen_topic \
    --invariant_regularization \
    --topic_words_path Amazon_topic_words.txt 
```

For other runing scripts, please refer to the folder `scripts`

## Usage:
```python
from ERLAS.model.erlas import ERLAS
from transformers import AutoTokenizer

model = ERLAS.from_pretrained('Hieuman/erlas')
tokenizer = AutoTokenizer.from_pretrained('Hieuman/erlas')

batch_size = 3
episode_length = 16
text = [
    ["Foo"] * episode_length,
    ["Bar"] * episode_length,
    ["Zoo"] * episode_length,
]
text = [j for i in text for j in i]
tokenized_text = tokenizer(
    text, 
    max_length=32,
    padding="max_length", 
    truncation=True,
    return_tensors="pt"
)
# inputs size: (batch_size, episode_length, max_token_length)
tokenized_text["input_ids"] = tokenized_text["input_ids"].reshape(batch_size, 1, episode_length, -1)
tokenized_text["attention_mask"] = tokenized_text["attention_mask"].reshape(batch_size, 1, episode_length, -1)

author_reps, _ = model(tokenized_text['input_ids'], tokenized_text['attention_mask'])

author_reps = author_reps.squeeze(1) # [bs, hidden_size]
```

## Citation 
```text
@inproceedings{10.1145/3626772.3657956,
author = {Man, Hieu and Huu Nguyen, Thien},
title = {Counterfactual Augmentation for Robust Authorship Representation Learning},
year = {2024},
isbn = {9798400704314},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3626772.3657956},
doi = {10.1145/3626772.3657956},
pages = {2347–2351},
numpages = {5},
keywords = {authorship attribution, counterfactual learning, domain generalization},
location = {Washington DC, USA},
series = {SIGIR '24}
}
```

## Bugs or questions?
If you have any questions about the code, feel free to open an issue on the GitHub repository.

