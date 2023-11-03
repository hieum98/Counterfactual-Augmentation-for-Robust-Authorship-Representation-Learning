import gc
from argparse import Namespace
import json
from math import ceil
from collections import defaultdict
import os
import pathlib
import random
import re
from typing import Any
from more_itertools import collapse
import numpy as np
import torch
import tqdm
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from retriv import SparseRetriever
import retriv

from .base_dataset import BaseDataset
from ..utils.tools import f_stem, padding


class HardRetrievalBatchDataset(BaseDataset):
    def __init__(self, params: Namespace, 
                 dataset_name: str,
                 split: str, 
                 seed: int=1741,
                 bm25_percentage: float=0.0,
                 dense_percentage: float=0.0) -> None:
        super().__init__(params, dataset_name)
        self.params = params
        self.dataset_name = dataset_name
        if dataset_name == 'Amazon':
            self.data_file = os.path.join(self.dataset_path, 'cleaned_data.jsonl')
            self.queries_file = os.path.join(self.dataset_path, 'validation_queries.jsonl')
            self.candidates_file = os.path.join(self.dataset_path, 'validation_targets.jsonl')
        elif dataset_name == 'PAN':
            self.data_file = os.path.join(self.dataset_path, 'cleaned_data.jsonl')
            self.queries_file = os.path.join(self.dataset_path, 'pan_21_queries_raw.jsonl')
            self.candidates_file = os.path.join(self.dataset_path, 'pan_21_targets_raw.jsonl')
        elif dataset_name == 'PAN20':
            self.data_file = os.path.join(self.dataset_path, 'cleaned_data.jsonl')
            self.queries_file = os.path.join(self.dataset_path, 'pan_20_queries_raw.jsonl')
            self.candidates_file = os.path.join(self.dataset_path, 'pan_20_targets_raw.jsonl')
        elif dataset_name == 'PAN21':
            self.data_file = os.path.join(self.dataset_path, 'cleaned_data.jsonl')
            self.queries_file = os.path.join(self.dataset_path, 'pan_21_queries_raw.jsonl')
            self.candidates_file = os.path.join(self.dataset_path, 'pan_21_targets_raw.jsonl')
        elif dataset_name == 'MUD':
            self.data_file = os.path.join(self.dataset_path, 'cleaned_data.jsonl')
            self.queries_file = os.path.join(self.dataset_path, 'test_queries.jsonl')
            self.candidates_file = os.path.join(self.dataset_path, 'test_targets.jsonl')
        else:
            raise "We haven't supported this dataset yet!"
            
        self.split = split
        self.text_key = 'syms'
        self.author_key = 'author_id'
        
        self.is_index_by_BM25 = params.index_by_BM25
        self.is_index_by_dense_retriever = params.index_by_dense_retriever
        
        if self.training_percentage < 1.0:
            preprocess_file = f'{self.training_percentage}_dr_{self.is_index_by_dense_retriever}_BM25_{self.is_index_by_BM25}_data.jsonl'
        else:
            preprocess_file = f'dr_{self.is_index_by_dense_retriever}_BM25_{self.is_index_by_BM25}_data.jsonl'
        preprocess_path = os.path.join(self.dataset_path, preprocess_file)
        full_preprocess_path = os.path.join(self.dataset_path, 'dr_True_BM25_True_data.jsonl')
        if (os.path.exists(preprocess_path) or os.path.exists(full_preprocess_path)) and self.split=='train' and (self.is_index_by_dense_retriever or self.is_index_by_BM25):
            preprocess_path = preprocess_path if os.path.exists(preprocess_path) else full_preprocess_path
            print(f"Loading preprocessed data from cache: {preprocess_path}")
            self.data = self.load_data(split=split, preprocess_path=preprocess_path)
            self.data = self.data.shuffle(seed=seed)
        else:
            print("Start preprocessing data ...")
            self.data = self.load_data(split=split)
            if self.is_index_by_BM25 and self.split=='train':
                # ELASTIC_PASSWORD = "XjOvVM+=yyheU_-7Ptj4"
                # self.es_client = Elasticsearch( "https://localhost:9200", 
                #                         ca_certs="/disk/hieu/elasticsearch-8.9.1/config/certs/http_ca.crt", 
                #                         basic_auth=("elastic", ELASTIC_PASSWORD),
                #                         request_timeout=10000000)
                self.index_by_BM25()
            if self.is_index_by_dense_retriever and self.split=='train':
                self.retrieval_encoder = AutoModel.from_pretrained(params.retriever_model)
                self.retrieval_tokenizer = AutoTokenizer.from_pretrained(params.retriever_model)
                self.retrieval_encoder = self.retrieval_encoder.cuda()
                retrieval_result = self.index_by_dense_retriever()
            
            if self.split=='train' and self.is_index_by_BM25 and self.is_index_by_dense_retriever:
                print("Saving cache.....")
                with open(preprocess_path, 'w', encoding='utf-8') as f:
                    for idx, data in tqdm.tqdm(enumerate(self.data)):
                        datapoint = {self.author_key: data[self.author_key],
                                     self.text_key: data[self.text_key],
                                     'dense_retriever_hard_example_idx': [int(item) for item in set(retrieval_result[idx])], 
                                     'BM25_hard_example_idx': [int(item) for item in set(data['BM25_hard_example_idx'])]}
                        json.dump(datapoint, f)
                        f.write('\n')
                self.data = self.data.map(lambda x, idx: {'dense_retriever_hard_example_idx': retrieval_result[idx]}, with_indices=True)
        self.batch_size = params.batch_size if params.batch_size < len(self.data) else len(self.data)
        self.bm25_percentage = bm25_percentage 
        self.dense_percentage = dense_percentage
        if self.split != 'train':
            self.token_max_length = 128

        if hasattr(self, 'topic_words'):
            self.augment_data()
        
    def load_data(self, split:str, preprocess_path=None):
        if split=='train':
            if preprocess_path != None:
                print(f"Loading data from {preprocess_path} ....")
                data = load_dataset('json', data_files=preprocess_path, split='train', cache_dir='cache')
            else:
                print(f"Loading data from {self.data_file} ....")
                data = load_dataset('json', data_files=self.data_file, split='train', cache_dir='cache')
                if self.training_percentage < 1.0:
                    print(f"Loading {self.training_percentage} data....")
                    data = data.train_test_split(train_size=self.training_percentage, load_from_cache_file=False)['train']
        else:
            print(f"Loading data from {self.queries_file}, and {self.candidates_file} ....")
            queries = load_dataset('json', data_files=self.queries_file, split='train', cache_dir='cache')
            queries = queries.map(lambda example: {'is_query': True})
            candidates = load_dataset('json', data_files=self.candidates_file, split='train', cache_dir='cache')
            candidates = candidates.map(lambda example: {'is_query': False})
            assert candidates.features.type == queries.features.type
            data = concatenate_datasets([queries, candidates])            
        return data
    
    def index_by_BM25(self):
        num_hard_example = 1000 if len(self.data) > 1000 else len(self.data)
        
        retriv.set_base_path("retriv")
        
        self.data = self.data.map(lambda example, idx: {"author_content": " ".join(example[self.text_key])[:5000], "id": idx}, with_indices=True)
        try:
            spare_retriver = SparseRetriever.load(f'{self.dataset_name}_{self.split}_{self.training_percentage}'.lower())
            print("index loaded!")
        except:
            self.data.to_json(f'cache/{self.dataset_name}_{self.split}.jsonl')
            spare_retriver = SparseRetriever(index_name=f'{self.dataset_name}_{self.split}_{self.training_percentage}'.lower(),
                                    model="bm25",
                                    min_df=1,
                                    tokenizer="whitespace",
                                    stemmer=None,
                                    stopwords=None,
                                    do_lowercasing=False,
                                    do_ampersand_normalization=False,
                                    do_special_chars_normalization=True,
                                    do_acronyms_normalization=False,
                                    do_punctuation_removal=False,)
            spare_retriver = spare_retriver.index_file(path=f'cache/{self.dataset_name}_{self.split}.jsonl',    # File kind is automatically inferred
                                                                show_progress=True,                                       # Default value
                                                                callback=lambda doc: {"id": doc["id"],                    # Callback defaults to None.
                                                                                    "text": doc["author_content"][:5000],}
                                                                )
            print("index computed!")
        
        def retrieve(batch):
            queries = [{'id': i, 'text': q} for i, q in enumerate(batch['author_content'])]
            retrieval_results = spare_retriver.msearch(
                queries=queries,
                cutoff=num_hard_example
            )
            return {'BM25_hard_example_idx': [list(item.keys()) for item in retrieval_results.values()],
                    self.author_key: batch[self.author_key],
                    self.text_key: batch[self.text_key]}

        self.data = self.data.map(lambda batch: retrieve(batch), batched=True, batch_size=500, 
                                  cache_file_name=f"cache/{self.training_percentage}_BM25_{self.dataset_name}.pyarow", 
                                  remove_columns=self.data.column_names)
        del spare_retriver
        gc.collect()
        
    def index_by_dense_retriever(self):
        num_hard_example = 1000 if len(self.data) > 1000 else len(self.data)
        def compute_embedding(example, rank, retrieval_encoder, retrieval_tokenizer):
            input_ids = retrieval_tokenizer(example["author_content"], 
                                            return_tensors="pt", 
                                            truncation=True,
                                            padding='longest', 
                                            max_length=512).input_ids.cuda()
            with torch.no_grad():
                emb = retrieval_encoder(input_ids)[0][:, 0]
            
            example['embeddings'] = emb.cpu().tolist()
            return example
        
        data_with_index = self.data.map(lambda example: {"author_content": " ".join(example[self.text_key])})
        data_with_index = data_with_index.map(lambda example, rank: compute_embedding(example, rank, self.retrieval_encoder, self.retrieval_tokenizer), 
                                              batched=True, batch_size=256, with_rank=True, 
                                              cache_file_name=f"cache/{self.training_percentage}_condenser_{self.dataset_name}.pyarow")
        data_with_index.add_faiss_index(column='embeddings', faiss_verbose=True)
        
        retrieval_result = []
        for example in tqdm.tqdm(data_with_index):
            query_embedding = torch.tensor(example['embeddings']).numpy()
            _, retrieval_idx = data_with_index.search('embeddings', query_embedding, k=num_hard_example)
            retrieval_result.append(list(set(retrieval_idx) - set([example['author_id']])))
        
        del data_with_index
        
        return retrieval_result
    
    def augment_data(self):
        def augment_batch(batch):
            author_docs = batch[self.text_key]
            author_data = {
                self.text_key: [],
                'topic_word_pos': []
            }
            for ori_docs in author_docs:
                num_docs = len(ori_docs)
                num_augmented = ceil((self.augmented_percentage / (1 - self.augmented_percentage)) * num_docs)
                tmp = []
                for i in range(num_augmented):
                    text = random.choice(ori_docs)
                    _text = self.augmented(text, mask_rate=0.5)
                    if _text != None:
                        tmp.append(_text)
                augmented_docs = ori_docs + tmp
                topic_word_pos = self.find_word_position(augmented_docs, self.topic_words)

                author_data[self.text_key].append(augmented_docs)
                author_data['topic_word_pos'].append(topic_word_pos)
            return author_data
        
        self.data = self.data.map(augment_batch, batched=True, batch_size=100, num_proc=20, 
                                  cache_file_name=f"cache/{self.training_percentage}_augmented_{self.augmented_percentage}_{self.dataset_name}.pyarow")

    def train_collate_fn(self, batch):
        """This function will sample a random number of episodes as per Section 2.3 of:
                https://arxiv.org/pdf/2105.07263.pdf
        """
        author = []
        input_ids, attention_mask, invariant_mask = [], [], []
        for item in batch:
            author.extend(item['author'])
            input_ids.extend(item['input_ids'])
            attention_mask.extend(item['attention_mask'])
            invariant_mask.extend(item['invariant_mask'])

        author_dict = {a_id: i for i, a_id in enumerate(set(collapse(author)))}
        author = [[author_dict[a] for a in item] for item in author]

        author = torch.tensor(author)

        # Minimum number of posts for an author history in batch
        min_posts = min([d.shape[1] for d in input_ids])

        # Size of episode = R + ⌈x(S-R)⌉, x ~ Beta(3,1)
        sample_size = min(1 + ceil(np.random.beta(3, 1) * self.params.episode_length), min_posts)

        # If minimum posts < episode length, make start 0 to ensure you get all posts
        if min_posts < self.params.episode_length:
            start = 0
        else:
            # Pick a random start index
            start = np.random.randint(0, self.params.episode_length - sample_size + 1)
    
        input_ids = torch.stack(padding([f[:, start:start + sample_size, :] for f in input_ids], pad_value=self.tokenizer.pad_token_id)) # (bs, a, d.p.a, l)
        attention_mask = torch.stack(padding([f[:, start:start + sample_size, :] for f in attention_mask], pad_value=0))
        invariant_mask = torch.stack(padding([f[:, start:start + sample_size, :] for f in invariant_mask], pad_value=0))
        data = [input_ids, attention_mask, invariant_mask]
        return data, author
    
    def val_test_collate_fn(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        author = [item['author'] for item in batch]
        is_query = [item['is_query'] for item in batch]
        input_ids = torch.stack(padding([f for f in input_ids], pad_value=self.tokenizer.pad_token_id))
        attention_mask = torch.stack(padding([f for f in attention_mask], pad_value=0))
        data = [input_ids, attention_mask] 
        return data, author, is_query

    def __len__(self):
        return 15*ceil(len(self.data)/self.batch_size) if self.split=='train' else len(self.data)
    
    def __getitem__(self, index) -> Any:
        if self.split=='train':
            num_BM25_hard_example = int(self.batch_size * self.bm25_percentage)
            num_dense_hard_example = int(self.batch_size * self.dense_percentage)
            
            author_data = self.data[index]
            if self.dense_percentage > 0 and self.is_index_by_dense_retriever:
                faiss_neighbor_ids = author_data['dense_retriever_hard_example_idx'][:num_dense_hard_example]
            else:
                faiss_neighbor_ids = []
            if self.bm25_percentage > 0 and self.is_index_by_BM25:
                es_neighbor_ids = author_data['BM25_hard_example_idx'][:num_BM25_hard_example]
            else:
                es_neighbor_ids = []
            neighbor_ids = list(set(faiss_neighbor_ids + es_neighbor_ids))
            retrieved_authors =  self.data[neighbor_ids]
            retrieved_authors =  [dict(zip(retrieved_authors, t)) for t in zip(*retrieved_authors.values())]
            
            data = [author_data, ]
            query_a_id = author_data[self.author_key]
            for ex in retrieved_authors:
                a_id = ex[self.author_key]
                if a_id != query_a_id:
                    data.append(ex)
            
            normal_examples = random.choices(range(len(self.data)), k=self.batch_size-len(data))
            data.extend([self.data[idx] for idx in normal_examples])
            
            author = []
            input_ids = []
            attention_mask = []
            invariant_mask = []
            for item in data:
                _input_ids, _attention_mask, _invariant_mask, _author = self.process_author_data(item)
                input_ids.append(_input_ids)
                attention_mask.append(_attention_mask)
                invariant_mask.append(_invariant_mask)
                author.append(_author)

            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'invariant_mask': invariant_mask, 'author': author}
        else:
            author_data = self.data[index]
            tokenized_episode = self.tokenizer(
                author_data[self.text_key][:64], 
                padding="max_length", 
                truncation=True, 
                max_length=self.token_max_length, 
                return_tensors='pt'
            )
            data = self.reformat_tokenized_inputs(tokenized_episode)
            
            data = [d.reshape(1, -1, self.token_max_length) for d in data]
            author = [author_data[self.author_key]]
            is_query = [author_data['is_query']]
            return {'input_ids': data[0], 'attention_mask': data[1], 'author': author, 'is_query': is_query}
        
        
        
        

