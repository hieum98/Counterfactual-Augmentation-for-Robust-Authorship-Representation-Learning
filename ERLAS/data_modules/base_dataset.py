from abc import ABC, abstractmethod
from argparse import Namespace
from copy import deepcopy
import copy
from math import ceil
import os
import random
import re
import torch
from typing import Any
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

from .constants import DATASET_PATH
from ..utils.tools import f_stem

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
        self.topic_words = []
        if os.path.exists(params.topic_words_path):
            print("Load topic words from {}".format(params.topic_words_path))
            with open(params.topic_words_path, 'r', encoding='utf-8') as f :
                for line in f:
                    line = line.strip()
                    if len(line) > 3:
                        self.topic_words.append(line)
    
    def tokenize_text(self, all_text, topic_word_pos=None):
        tokenized_episode = self.tokenizer(
            all_text, 
            padding="longest", 
            truncation=True, 
            max_length=self.token_max_length, 
            return_tensors='pt'
        )
        if topic_word_pos != None:
            assert len(topic_word_pos) == tokenized_episode['attention_mask'].size(0)
            topic_mask = torch.ones_like(tokenized_episode['attention_mask'])
            for i in range(tokenized_episode['attention_mask'].size(0)):
                for char_pos in topic_word_pos[i]:
                    word_pos = tokenized_episode.char_to_word(i, char_pos)
                    if word_pos != None:
                        _token_pos = tokenized_episode.word_to_tokens(i, word_pos)
                        for token_idx in range(_token_pos[0], _token_pos[1]):
                            topic_mask[i, token_idx] = 0

            tokenized_episode =  self.reformat_tokenized_inputs(tokenized_episode, topic_mask)
        else:
            tokenized_episode =  self.reformat_tokenized_inputs(tokenized_episode)
        
        return tokenized_episode

    def reformat_tokenized_inputs(self, tokenized_episode, topic_mask=None):
        """Reformats the output from HugginFace Tokenizers.
        """
        if len(tokenized_episode.keys()) == 3:
            input_ids, _, attention_mask = tokenized_episode.values()
        else:
            input_ids, attention_mask = tokenized_episode.values()
        mask_index = torch.nonzero(torch.where(input_ids == self.tokenizer.mask_token_id, torch.tensor(self.tokenizer.mask_token_id), torch.tensor(0)))
        attention_mask.index_put_(tuple(mask_index.t()), torch.tensor([0]))

        if topic_mask == None:
            data = [input_ids, attention_mask]
        else:
            data = [input_ids, attention_mask, attention_mask*topic_mask]

        return data
    
    def augmented(self, text: str, mask_rate=0.5):
        tokenized_text = text.split(' ')
        n_t = len(tokenized_text)
        tokenized_idx = range(n_t)
        topic_token_idx = []
        for idx, token in zip(tokenized_idx, tokenized_text):
            token = token.lower().strip()
            token = re.sub('[^a-z]', '', token)
            if len(token) != 0 and f_stem([token])[0] in self.topic_words:
                topic_token_idx.append(idx)
        
        if len(topic_token_idx) <= 0:
            return None
        
        n_mask = ceil(mask_rate * len(topic_token_idx))
        random.shuffle(topic_token_idx)
        mask_id = topic_token_idx[:n_mask]
        _text = deepcopy(tokenized_text)
        for idx in mask_id:
            _text[idx] = self.tokenizer.mask_token
        _text = ' '.join(_text)
        return _text
    
    def find_word_position(self, docs: list[str], words_list: list[str]):
        word_pos = []
        for text in docs:
            tokenized_text = text.split(' ')
            _word_pos = []
            start = 0
            for token in tokenized_text:
                _token = token.lower().strip()
                _token = re.sub('[^a-z]', '', _token)
                if len(_token) != 0 and f_stem([_token])[0] in words_list:
                    _word_pos.append(start)
                start = start + len(token) + 1
            word_pos.append(_word_pos)
        return word_pos
    
    def sample_episode(self, author_data, is_test=False):
        """Samples a episode of size `episode_length`.
        Args:
            index (int): Index of the author we're sampling.
        """
        author_data = copy.deepcopy(author_data)
        
        num_docs = len(author_data[self.text_key])
        episode_length = self.episode_length
        docs = author_data[self.text_key]
        topic_word_pos = author_data['topic_word_pos']
        if num_docs < self.episode_length:
            num_augmented = self.episode_length - num_docs
            for i in range(num_augmented):
                idx = random.choice(range(num_docs))
                _text = docs[idx]
                _topic_word_pos = topic_word_pos[idx]
                docs.append(_text)
                topic_word_pos.append(_topic_word_pos)
            num_docs = episode_length
        
        temp  = list(zip(docs, topic_word_pos))
        random.shuffle(temp)
        docs, topic_word_pos = zip(*temp)

        maxval = num_docs - episode_length

        if is_test:
            start_index = 0
        else:
            start_index = random.randint(0, maxval)
    
        episode = {self.text_key: docs[start_index: start_index + episode_length],
                   'topic_word_pos': topic_word_pos[start_index: start_index + episode_length]}
         
        episode['author_id'] = author_data[self.author_key]
            
        return episode
    
    def process_author_data(self, author_data):
        text = []
        author = []
        topic_word_pos = []
        for _ in range(self.num_sample_per_author):
            episode = self.sample_episode(author_data)
            text.extend(episode[self.text_key])
            topic_word_pos.extend(episode['topic_word_pos'])
            author.append(episode["author_id"])

        input_ids, attention_mask, invariant_mask = self.tokenize_text(text, topic_word_pos)
        input_ids = input_ids.reshape(self.num_sample_per_author, -1, attention_mask.size(-1)) # (a, d.p.a, l)
        attention_mask = attention_mask.reshape(self.num_sample_per_author, -1, attention_mask.size(-1))
        invariant_mask = invariant_mask.reshape(self.num_sample_per_author, -1, attention_mask.size(-1))
        return input_ids, attention_mask, invariant_mask, author
    
    @abstractmethod
    def load_data(self, cache_dir: str):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, index) -> Any:
        raise NotImplementedError

