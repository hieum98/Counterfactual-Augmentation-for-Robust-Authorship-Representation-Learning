
import argparse
import time


def create_argument_parser():
    """Defines a parameter parser for all of the arguments of the application.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    ##### Data Hyperparameters #####
    parser.add_argument("--dataset_name", nargs='+', type=str,
                        help="Datasets to use")
    parser.add_argument("--training_percentage", type=float, default=1.0,
                        help="The effective of training size")
    parser.add_argument("--episode_length", type=int, default=16,
                        help="Number of actions to include in an episode")
    parser.add_argument("--token_max_length", type=int, default=32,
                        help="Number of tokens to take per example")
    parser.add_argument("--num_sample_per_author", type=int, default=2,
                        help="Number of episodes to sample per author during training")
    parser.add_argument("--index_by_BM25", action='store_true', default=False,
                        help="Whether or not to index by BM25 on the training set")
    parser.add_argument("--index_by_dense_retriever", action='store_true', default=False,
                        help="Whether or not to index by dense retriver model on the training set")
    parser.add_argument("--retriever_model", type=str, default="Luyu/condenser",
                        help="Specifies which dense retriever to use")
    parser.add_argument("--BM25_percentage", type=float, default=0.5,
                        help="The proportion of hard examples retrieverd by elasticsearch")
    parser.add_argument("--dense_percentage", type=float, default=0.5,
                        help="The proportion of hard examples retrieverd by faiss")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Number of authors to include in each batch")
    parser.add_argument("--augmented_percentage", type=int, default=0.0,
                        help="The percentage of augmented episode in a batch")
    
    ##### Model Hyperparameters #####
    parser.add_argument("--model_type", type=str, default="distilroberta-base",
                        help="Specifies which Transformer backbone to use")
    parser.add_argument("--embedding_dim", type=int, default=512,
                        help="Final output embedding dimension")
    
    ##### Training Hyperparameters #####
    parser.add_argument("--gradient_checkpointing", default=False, action="store_true",
                        help="If True, activates Gradient Checkpointing")
    parser.add_argument("--miner_type", type=str, default="",
                        help="Specifies which miner to use")  
    parser.add_argument("--loss_type", type=str, default="",
                        help="Specifies which loss to use")  
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to use for training")
    parser.add_argument("--use_gc", default=False, action="store_true", 
                        help="Use GradCache for memory-efficient large-batch contrastive learning",)
    parser.add_argument("--gc_minibatch_size", type=int, default=4, 
                        help="the small batch size as specified in GradCache",)
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Specifies learning rate")
    parser.add_argument("--learning_rate_scaling", action="store_true", default=False,
                        help="Toggles variance-based learning rate scaling")
    parser.add_argument("--temperature", type=float, default=0.01,
                        help="Temperature to use for SupCon")
    parser.add_argument("--num_epoch", type=int, default=20,
                        help="Number of epochs")
    
    ##### MISC #####
    parser.add_argument("--precision", default='16-mixed', type=str,
                        help="Precision of model weights")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers to prefetch data")
    parser.add_argument("--pin_memory", action='store_true', default=False,
                        help="Used pin memory for prefetching data")
    parser.add_argument("--random_seed", type=int, default=1741,
                        help="Seed for PyTorch and NumPy random operations")
    parser.add_argument('--output_path', type=str, default='experiments',
                        help="Location of training output")
    parser.add_argument("--experiment_id", type=str, default="{}".format(int(time.time()),
                        help="Experiment identifier for an experiment group"))
    parser.add_argument("--version", type=str, default=None,
                        help="PyTorch Lightning's folder version name.")
    parser.add_argument("--load_checkpoint", default=None, type=str,
                        help="Location of model checkpoint")
    parser.add_argument("--do_learn", action='store_true', default=False,
                        help="Whether or not to train on the training set")
    parser.add_argument("--evaluate", action='store_true', default=False,
                        help="Whether or not to evaluate on the test set")
    
    return parser.parse_args()
    
    