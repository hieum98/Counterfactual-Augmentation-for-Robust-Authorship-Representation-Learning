from argparse import Namespace
import os
from typing import Any
import torch
from torch.utils.data.dataset import Dataset
from datasets import load_dataset
from tokenizers import Tokenizer, normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from .constants import DATASET_PATH
from ..utils.tools import preprocess


class LDADataset(Dataset):
    def __init__(self, params: Namespace, 
                 dataset_name: str) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_path = DATASET_PATH[dataset_name]
        if dataset_name == 'Amazon':
            self.data_file = os.path.join(self.dataset_path, 'cleaned_data.jsonl')
        elif dataset_name == 'PAN':
            self.data_file = os.path.join(self.dataset_path, 'cleaned_data.jsonl')
        elif dataset_name == 'PAN20':
            self.data_file = os.path.join(self.dataset_path, 'cleaned_data.jsonl')
        elif dataset_name == 'PAN21':
            self.data_file = os.path.join(self.dataset_path, 'cleaned_data.jsonl')
        elif dataset_name == 'MUD':
            self.data_file = os.path.join(self.dataset_path, 'cleaned_data.jsonl')
        else:
            raise "We haven't supported this dataset yet!"
        self.training_percentage = params.training_percentage
        self.text_key = 'syms'
        self.author_key = 'author_id'

        self.data = load_dataset('json', data_files=self.data_file, split='train', cache_dir='cache')
        if self.training_percentage < 1.0:
            print(f"Loading {self.training_percentage} data....")
            self.data = self.data.shuffle()
            self.data = self.data.train_test_split(train_size=self.training_percentage, load_from_cache_file=False)['train']
        
        self.preprocess()
        
        tokenizer_file = f"{self.dataset_name}_{self.training_percentage}_tokenizer.json"
        tokenizer_path = os.path.join(self.dataset_path, tokenizer_file)
        if os.path.exists(tokenizer_path):
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
            self.tokenizer.pad_token = "[PAD]"
        else:
            tokenizer = self.train_tokenizer()
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
            self.tokenizer.pad_token = "[PAD]"
            tokenizer.save(tokenizer_path)
        
        self.vocab_size = len(self.tokenizer)

    def preprocess(self):
        def preprocess_batch(batch):
            docs = []
            for item in batch[self.text_key]:
                docs.extend(item)
            
            cleaned_docs = []
            for item in docs:
                cleaned_item = preprocess(item)
                if len(cleaned_item) > 0:
                    cleaned_docs.append(' '.join(cleaned_item))
            
            return {'cleaned_doc': cleaned_docs}
                
        self.data = self.data.map(preprocess_batch, batched=True, batch_size=100,
                            cache_file_name=f"cache/{self.training_percentage}_LDA_preprocess_{self.dataset_name}.pyarow", 
                            num_proc=32, remove_columns=self.data.column_names)

    def train_tokenizer(self):
        def batch_iterator(batch_size=1000):
            for i in range(0, len(self.data), batch_size):
                yield self.data[i : i + batch_size]["cleaned_doc"]

        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
        trainer = WordLevelTrainer(vocab_size=50000, min_frequency=5, special_tokens=["[UNK]", "[PAD]"])
        tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(self.data))
        return tokenizer
    
    def train_collate_fn(self, batch):
        bs = len(batch)
        _input_ids = self.tokenizer(batch, return_tensors='pt', padding=True).input_ids # (bs, max_len)
        input_ids = torch.zeros((bs, self.vocab_size)) 
        for i in range(bs):
            for idx in _input_ids[i]:
                if idx != self.tokenizer.pad_token_id and idx != self.tokenizer.unk_token_id:
                    input_ids[i, idx] = input_ids[i, idx] + 1
        return input_ids 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> Any:
        return self.data[index]['cleaned_doc']
