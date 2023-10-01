
from datetime import timedelta
import glob
import os
import random

import numpy as np
import pytorch_lightning as pt
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from ERLAS.aurguments import create_argument_parser
from ERLAS.model.lightning_trainer import LightningTrainer

torch.backends.cuda.matmul.allow_tf32 = True


def main(params):
    # set random seeds reproducibility
    random.seed(params.random_seed)
    np.random.seed(params.random_seed)
    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed(params.random_seed)

    # weirdness with HuggingFace tokenizer when processing things in parallel
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.multiprocessing.set_sharing_strategy('file_system')

    # create experiment_dir and load model
    experiment_dir = os.path.join(params.output_path, params.experiment_id)
    model = LightningTrainer(params)
    # train_dataloader = model.train_dataloader()

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(experiment_dir, params.version, 'checkpoints'),
        filename="step={step}",
        auto_insert_metric_name=False,
        monitor=None,
        save_top_k=-1,
        every_n_train_steps=500,
    )

    # load checkpoint if necessary
    resume_from_checkpoint = None
    if params.load_checkpoint != None:
        # get the latest checkpoint
        resume_from_checkpoint = params.load_checkpoint
        print(f"Checkpoint: {resume_from_checkpoint}")

        checkpoint = torch.load(resume_from_checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    logger = TensorBoardLogger(experiment_dir, version=params.version)
    lr_logger = LearningRateMonitor(logging_interval='step')
    if params.do_learn:
        trainer = pt.Trainer(default_root_dir=experiment_dir, 
                            max_epochs=params.num_epoch,
                            logger=logger,
                            enable_checkpointing=True,
                            accelerator='gpu', 
                            devices=params.gpus,
                            strategy='ddp_find_unused_parameters_true' if params.gpus > 1 else 'auto',
                            precision=params.precision,
                            reload_dataloaders_every_n_epochs=1,
                            callbacks = [checkpoint_callback, lr_logger,],)
        trainer.fit(model)
    if params.evaluate:
        trainer = pt.Trainer(default_root_dir=experiment_dir, 
                            logger=logger,
                            enable_checkpointing=True,
                            accelerator='gpu', 
                            devices=[params.gpus],
                            strategy='auto',
                            precision=params.precision,
                            callbacks = [checkpoint_callback, lr_logger,],)
        trainer.test(model)

if __name__ == "__main__":
    params = create_argument_parser()
    main(params)
