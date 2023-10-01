from abc import ABC, abstractmethod
from argparse import Namespace
from copy import deepcopy
import copy
from math import ceil
import random
from typing import Any
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

from .constants import DATASET_PATH

class BaseDataset(ABC, Dataset):
    def __init__(self, params: Namespace, 
                 dataset_name: str,) -> None:
        super().__init__()
        self.dataset_path = DATASET_PATH[dataset_name]
        self.training_percentage = params.training_percentage
        self.num_sample_per_author = params.num_sample_per_author
        self.episode_length = params.episode_length
        self.token_max_length = params.token_max_length
        self.augmented_percentage = params.augmented_percentage
        
        self.tokenizer = AutoTokenizer.from_pretrained(params.model_type)
    
    def tokenize_text(self, all_text):
        tokenized_episode = self.tokenizer(
            all_text, 
            padding="longest", 
            truncation=True, 
            max_length=self.token_max_length, 
            return_tensors='pt'
        )
        tokenized_episode =  self.reformat_tokenized_inputs(tokenized_episode)
        
        return tokenized_episode

    def reformat_tokenized_inputs(self, tokenized_episode):
        """Reformats the output from HugginFace Tokenizers.
        """
        if len(tokenized_episode.keys()) == 3:
            input_ids, _, attention_mask = tokenized_episode.values()
            data = [input_ids, attention_mask]
        else:
            input_ids, attention_mask = tokenized_episode.values()
            data = [input_ids, attention_mask]

        return data
    
    def augmented(self, text, mask_rate=0.15):
        tokenized_text = text.split(' ')
        n_t = len(tokenized_text)
        tokenized_idx = range(n_t)
        len_token = [len(t) for t in tokenized_text]
        non_space_token_idx = []
        for idx, l in zip(tokenized_idx, len_token):
            if l != 0:
                non_space_token_idx.append(idx)
        n_mask = ceil(mask_rate * n_t)
        random.shuffle(non_space_token_idx)
        mask_id = non_space_token_idx[:n_mask]
        _text = deepcopy(tokenized_text)
        for idx in mask_id:
            _text[idx] = self.tokenizer.mask_token
        _text = ' '.join(_text)
        return _text
    
    def sample_episode(self, author_data, is_test=False):
        """Samples a episode of size `episode_length`.
        Args:
            index (int): Index of the author we're sampling.
        """
        author_data = copy.deepcopy(author_data)
        
        num_docs = len(author_data[self.text_key])
        episode_length = self.episode_length
        docs = author_data[self.text_key]
        if num_docs < self.episode_length:
            num_augmented = self.episode_length - num_docs
            tmp = []
            tmp_origin = []
            for i in range(num_augmented):
                text = random.choice(docs)
                _text = self.augmented(text, mask_rate=0.2)
                tmp.append(_text)
                tmp_origin.append(text)
            author_data[self.text_key].extend(tmp)
            num_docs = episode_length
        
        num_augmented = (self.augmented_percentage / (1 - self.augmented_percentage)) * num_docs
        tmp = []
        tmp_origin = []
        for i in range(num_augmented):
            text = random.choice(docs)
            _text = self.augmented(text, mask_rate=0.2)
            tmp.append(_text)
            tmp_origin.append(text)
        author_data[self.text_key].extend(tmp)
        
        random.shuffle(author_data[self.text_key])

        maxval = num_docs - episode_length

        if is_test:
            start_index = 0
        else:
            start_index = random.randint(0, maxval)
    
        episode = {k: v[start_index: start_index + episode_length] 
                for k, v in author_data.items() if isinstance(v, list) and len(v) > 0}
         
        episode['author_id'] = author_data[self.author_key]
            
        return episode
    
    def process_author_data(self, author_data):
        text = []
        author = []
        for _ in range(self.num_sample_per_author):
            episode = self.sample_episode(author_data)
            text.extend(episode[self.text_key])
            author.append(episode["author_id"])

        input_ids, attention_mask = self.tokenize_text(text)
        input_ids = input_ids.reshape(self.num_sample_per_author, -1, attention_mask.size(-1)) # (a, d.p.a, l)
        attention_mask = attention_mask.reshape(self.num_sample_per_author, -1, attention_mask.size(-1))
        return input_ids, attention_mask, author
    
    @abstractmethod
    def load_data(self, cache_dir: str):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, index) -> Any:
        raise NotImplementedError

