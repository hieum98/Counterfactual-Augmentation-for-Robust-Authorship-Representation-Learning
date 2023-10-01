
import copy
import os
from abc import ABC, abstractmethod
from contextlib import nullcontext
from itertools import chain
from typing import Any, Optional
from more_itertools import collapse
from math import ceil
from einops import rearrange
from matplotlib import pyplot as plt

import numpy as np
import pytorch_lightning as pt
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
import torch
import torch.optim as optim
from torch.cuda.amp import autocast
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils import distributed as pml_dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .erlas import ERLAS
from ..data_modules.hard_retrieval_batch_dataset import HardRetrievalBatchDataset
from ..utils.combined_loader import CombinedLoader
from ..utils.metric import compute_metrics


class LightningTrainer(pt.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()
        self.params = params
        
        self.model = ERLAS(params)
        
        self.learning_rate = params.learning_rate
        if self.params.miner_type == 'multi-similariry':
            print("Use Multi Similarity Miner")
            self._miner = miners.MultiSimilarityMiner(epsilon=0.1)
        if self.params.loss_type == 'multi-similariry':
            print("Use Multi Similarity Loss")
            self._loss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
        else:
            print("Use Supervised Constrastive Loss")
            self._loss = losses.SupConLoss(
                temperature=self.params.temperature, 
                distance=CosineSimilarity()
            )
        if params.gpus > 1:
            self.loss = pml_dist.DistributedLossWrapper(self._loss)
        else:
            self.loss =self._loss
        if hasattr(self, 'miner') and params.gpus > 1:
            self.miner = pml_dist.DistributedMinerWrapper(self._miner)
        else:
            self.loss =self._loss

        self.automatic_optimization = not self.params.use_gc
        self.fp16 = self.params.precision in ['16', '16-mixed']
        self.test_queries = []
        self.test_candidates = []

    def init_gc(self, scaler, ddp_module):
        """Sets up the required components of GradCache
        This method is called after the model is initialized.
        """
        self.scaler = scaler
        if self.fp16 and self.params.use_gc:
            # pytorch lightning autocast wraps everything in it
            # it needs to be disabled in gradcache because we do forward twice, and one with no grad
            # then we do autocast manually in gradcache when we need to
            # original post: https://discuss.pytorch.org/t/autocast-and-torch-no-grad-unexpected-behaviour/93475/3?u=jxmorris12
            # pl source code: luar/vluar/lib/python3.8/site-packages/pytorch_lightning/plugins/precision/native_amp.py::forward_context
            self.trainer.strategy.precision_plugin.forward_context = nullcontext
        
        if not self.params.use_gc:
            # no-op
            return

        print(f"initializing gc with ddp_module={type(ddp_module)}")

        from ..utils import pt_gradcache

        print(
            f"** LuarLightningModule using gradcache with minibatch_size={self.params.gc_minibatch_size}"
        )
        self.gc = pt_gradcache.GradCache(
            models=[ddp_module],
            chunk_sizes=self.params.gc_minibatch_size,
            loss_fn=self.calculate_loss,
            get_rep_fn=(
                lambda x: x[0]
            ),  # (episode_embeddings, comment_embeddings) -> episode_embeddings
            fp16=self.fp16,
            scaler=(scaler if self.fp16 else None),
            backward_fn=self.manual_backward,
        )
    
    def calculate_loss(self, episode_embeddings, labels):
        """Calculate the customized model loss."""
        episode_embeddings = rearrange(episode_embeddings, "b n l -> (b n) l")
        if self.trainer.training:
            if hasattr(self, 'miner'):
                hard_pairs = self.miner(episode_embeddings, labels)
                return self.loss(episode_embeddings, labels, hard_pairs)
            else:
                return self.loss(episode_embeddings, labels)
        else:
            if hasattr(self, 'miner'):
                hard_pairs = self._miner(episode_embeddings, labels)
                return self._loss(episode_embeddings, labels, hard_pairs)
            else:
                return self._loss(episode_embeddings, labels)
            
    def configure_optimizers(self):
        """Configures the LR Optimizer & Scheduler.
        """
        # Scale learning rate to preserve variance based on success of 2e-5@64 per GPU
        if self.params.learning_rate_scaling:
            lr_factor = np.power(self.params.batch_size / 32, 0.5)
        else:
            lr_factor = 1
            
        learning_rate = self.learning_rate * lr_factor
        print("Using LR: {}".format(learning_rate))
        optimizer = optim.AdamW(
            chain(self.parameters()), lr=learning_rate, eps=1e-6)

        return [optimizer]
    
    def train_dataloader(self):
        """Returns the training DataLoader.
        """  
        datasets = {}      
        for name in self.params.dataset_name:
            dataset = HardRetrievalBatchDataset(self.params, 
                                                dataset_name=name, 
                                                split='train',
                                                seed=self.params.random_seed + self.current_epoch,
                                                bm25_percentage=self.params.BM25_percentage,
                                                dense_percentage=self.params.dense_percentage)
            datasets[name] = dataset
        
        print(f"Data infor: \n{[f'{name}: {data.__len__()}; ' for name, data in datasets.items()]}")

        data_loaders = {}
        for source, dataset in datasets.items():
            data_loaders[source] = DataLoader(dataset,
                                            batch_size=1,
                                            shuffle=True,
                                            num_workers=self.params.num_workers,
                                            pin_memory=self.params.pin_memory,
                                            collate_fn=dataset.train_collate_fn)
            
        return CombinedLoader(iterables=data_loaders, mode="random")
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        datasets = {} 
        for name in self.params.dataset_name:
            dataset = HardRetrievalBatchDataset(self.params, 
                                                dataset_name=name, 
                                                split='test',)
            datasets[name] = dataset
        
        print(f"Data infor: \n{[f'{name}: {data.__len__()}; ' for name, data in datasets.items()]}")

        data_loaders = {}
        for source, dataset in datasets.items():
            data_loaders[source] = DataLoader(dataset,
                                            batch_size=1,
                                            shuffle=True,
                                            num_workers=self.params.num_workers,
                                            pin_memory=self.params.pin_memory,
                                            collate_fn=dataset.val_test_collate_fn)
        return list(data_loaders.values())

    ##################################################
    # Training Methods
    ##################################################
    
    def forward(self, *data):
        """Calculates a fixed-length feature vector for a batch of episode samples."""
        output = self.model.get_episode_embeddings(data)
        return output
    
    def _model_forward(self, batch):
        """Passes a batch of data through the model.
        This is used in the lightning_trainer.py file.
        """
        data, labels = batch

        episode_embeddings, comment_embeddings = self.model.forward(*data)
        labels = torch.flatten(labels)

        return episode_embeddings, comment_embeddings
    
    def _internal_step(self, batch):
        """Internal step function used by training_step 
           functions.
        """
        labels = batch[-1].flatten()
        with autocast() if self.fp16 and self.params.use_gc else nullcontext():
            episode_embeddings, comment_embeddings = self._model_forward(batch)

        return_dict = {
            f"embedding": episode_embeddings,
            "ground_truth": labels,
        }

        return return_dict

    def on_train_start(self):
        self.init_gc(self.trainer.scaler, self.trainer.strategy.model)

    def training_step(self, *args, **kwargs):
        """Executes one training step."""
        if len(args) == 2 and isinstance(args[1], int):
            batch, batch_idx = args
            if len(batch) == 3:
                batch, _, _ = batch
        else:
            # Lightning DDP will forward calls to forward back to `_internal_step`,
            # so if DDP is enabled we have to intercept these calls and send them
            # back to forward() to get around restrictions of the lightning API.

            # If Lightning DDP uses strategy "nccl", we can do `self.init_gc(self.trainer.scaler, self.trainer.lightning_module)`
            # in `on_train_start(self)`, which refers to the base transformer model, then this part is never called.
            # If Lightning DDP does not have strategy "nccl", use `self.init_gc(self.trainer.scaler, self.trainer.strategy.model)` instead,
            # which will call this line.
            return self.forward(*args, **kwargs)
        if self.params.use_gc:
            assert self.gc is not None
            data, labels = batch
            # label = [batch_size, (2: query, target)]
            # data = (2: input_ids, attention_mask), [batch_size, (2: query, target), 16, 32]
            batch_size = labels.size(0)
            optimizer = self.optimizers()
            optimizer.zero_grad()
            loss = (
                self.gc(data, no_sync_except_last=(self.params.gpus > 1), labels=labels.flatten(),)
                / self.params.gpus
            )
            optimizer.step()

            step_outputs = {
                "loss": loss,
                "batch_size": batch_size,
            }
        else:
            step_outputs = self._internal_step(batch)
            
        if not self.params.use_gc:
            episode_embeddings = step_outputs["embedding"]
            labels = step_outputs["ground_truth"]

            # training_step_end is not handled by pl context manager, need to autocast manually
            with autocast() if self.fp16 else nullcontext():
                loss = self.calculate_loss(episode_embeddings, labels)
                
            return_dict = {"loss": loss}
            self.log("loss", loss,  prog_bar=True, sync_dist=True)
        else:
            self.log("loss", step_outputs["loss"],  prog_bar=True, sync_dist=True)
            return_dict = {"loss": step_outputs["loss"]}
        return return_dict
    
    def test_step(self, batch, batch_idx, dataloader_idx=0) -> STEP_OUTPUT | None:
        data, author, is_query = batch
        labels = torch.tensor(author)
        episode_embeddings, comment_embeddings = self.forward(*data)
        return_dict = {f"embedding": episode_embeddings,
                       "ground_truth": labels,}
        if is_query[0][0]:
            self.test_queries.append(return_dict)
        else:
            self.test_candidates.append(return_dict)
    
    def on_test_epoch_end(self) -> None:
        metrics = compute_metrics(self.test_queries, self.test_candidates)
        self.log_dict(metrics)
        self.test_queries.clear()
        self.test_candidates.clear()
