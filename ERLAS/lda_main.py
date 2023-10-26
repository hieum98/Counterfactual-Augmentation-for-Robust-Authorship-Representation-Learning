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
from ERLAS.data_modules.lda_dataset import LDADataset
from ERLAS.model.lda_trainer import LDATrainer

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
    dataset = LDADataset(params, dataset_name=params.dataset_name[0])
    vocab = dataset.tokenizer.vocab
    idx2word = {v: k for k, v in vocab.items()}
    vocab_size = dataset.vocab_size
    model = LDATrainer(params=params,
                       vocab_size=vocab_size,
                       num_topics=50,
                       lr=1e-3)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(experiment_dir, params.version, 'checkpoints'),
        filename="step={step}",
        auto_insert_metric_name=False,
        monitor=None,
        save_top_k=-1,
        every_n_epochs=1,
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
    trainer = pt.Trainer(default_root_dir=experiment_dir, 
                        max_epochs=params.num_epoch,
                        logger=logger,
                        accelerator='gpu', 
                        devices=params.gpus,
                        strategy='ddp_find_unused_parameters_true' if params.gpus > 1 else 'auto',
                        callbacks = [checkpoint_callback, lr_logger,],)
    trainer.fit(model)

    topics = model.top_words(idx2word=idx2word, n_words=100)
    with open(f'{params.dataset_name[0]}_topic_words.txt', 'w', encoding='UTF-8') as f: 
        for item in topics:
            f.write('\n'.join(item))
            f.write('\n')


if __name__ == "__main__":
    params = create_argument_parser()
    main(params)
    

