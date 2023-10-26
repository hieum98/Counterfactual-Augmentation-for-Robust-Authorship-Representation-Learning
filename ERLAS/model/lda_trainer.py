from argparse import Namespace
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS, OptimizerLRScheduler
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import LogNormal, Dirichlet
from torch.distributions import kl_divergence
import pytorch_lightning as pt

from .prod_lda import ProdLDA
from ..data_modules.lda_dataset import LDADataset
from ..utils.combined_loader import CombinedLoader


class LDATrainer(pt.LightningModule):
    def __init__(self, 
                 params: Namespace,
                 vocab_size: int,
                 num_topics: int, 
                 hidden_size: int=512,
                 use_lognormal: bool= False,
                 batch_size: int=256,
                 lr: float= 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.params = params

        self.model = ProdLDA(vocab_size=vocab_size, 
                             hidden_size=hidden_size,
                             num_topics=num_topics,
                             use_lognormal=use_lognormal,
                             dropout=0.2)
        self.lr = lr
        self.batch_size = batch_size

    def train_dataloader(self) -> TRAIN_DATALOADERS:   
        dataset = LDADataset(self.params, dataset_name=self.params.dataset_name[0])

        print(f"Data infor: \n {self.params.dataset_name[0]}: {dataset.__len__()}")

        data_loader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=20,
                                collate_fn=dataset.train_collate_fn)
            
        return data_loader
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,)
        return optimizer
    
    def compute_loss(self, inputs, outputs, posterior):
        def recon_loss(targets, outputs):
            nll = - torch.sum(targets * outputs)
            return nll

        def standard_prior_like(posterior):
            if isinstance(posterior, LogNormal):
                loc = torch.zeros_like(posterior.loc)
                scale = torch.ones_like(posterior.scale)        
                prior = LogNormal(loc, scale)
            elif isinstance(posterior, Dirichlet):
                alphas = torch.ones_like(posterior.concentration)
                prior = Dirichlet(alphas)
            return prior
        
        prior = standard_prior_like(posterior)
        nll = recon_loss(inputs, outputs)
        kld = torch.sum(kl_divergence(posterior, prior))
        return (nll + kld) / inputs.size(0)
    
    def forward(self, inputs) -> Any:
        return self.model(inputs)
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        outputs, posterior = self.model(batch)
        loss = self.compute_loss(batch, outputs, posterior)
        self.log('loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    def top_words(self, idx2word, n_words=100):
        beta = self.model.decode.fc.weight.cpu().detach().numpy().T
        topics = []
        for i in range(len(beta)):
            topic = [idx2word[j] for j in beta[i].argsort()[:-n_words-1:-1]]
            topics.append(topic)

        return topics


            
