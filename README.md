# Counterfactual Augmentation for Robust Authorship Representation Learning

This repository contains the code for the paper "Counterfactual Augmentation for Robust Authorship Representation Learning" (SIGIR 2024). 

## Install envrionment

Install evironment from ```requirements.txt```

Install the package

```
pip install -e .
```

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
