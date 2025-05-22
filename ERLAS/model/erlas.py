from argparse import Namespace
from functools import partial

from huggingface_hub import PyTorchModelHubMixin
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from transformers import AutoModel, T5EncoderModel

from .layers import SelfAttention


class ERLAS(nn.Module, PyTorchModelHubMixin):
    def __init__(self, params: Namespace = None, embedding_dim: int = None, model_type: str = None, gradient_checkpointing: bool = False):
        super().__init__()
        if params is None:
            params = Namespace(embedding_dim=embedding_dim, model_type=model_type, gradient_checkpointing=gradient_checkpointing)
        self.params = params
        self.create_transformer()
        
        self.attn_fn = SelfAttention()
        self.linear = nn.Linear(self.hidden_size, self.params.embedding_dim)
        
    def create_transformer(self):
        """Creates the Transformer model.
        """
        modelname = self.params.model_type
        self.transformer = AutoModel.from_pretrained(modelname, cache_dir='pretrained_weights')

        self.hidden_size = self.transformer.config.hidden_size
        self.num_attention_heads = self.transformer.config.num_attention_heads
        self.dim_head = self.hidden_size // self.num_attention_heads
        
        if self.params.gradient_checkpointing:
            self.transformer.gradient_checkpointing_enable()
        
    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean Pooling as described in the SBERT paper."""
        input_mask_expanded = repeat(
            attention_mask, "b l -> b l d", d=self.hidden_size
        ).float()
        sum_embeddings = reduce(
            token_embeddings * input_mask_expanded, "b l d -> b d", "sum"
        )
        sum_mask = torch.clamp(
            reduce(input_mask_expanded, "b l d -> b d", "sum"), min=1e-9
        )
        return sum_embeddings / sum_mask

    def get_episode_embeddings(self, data):
        """Computes the Author Embedding."""
        # b=batch_size=32, n=num_sample_per_author=1 (inference) or 2 (training), e=episode_length=16, t=token_max_length=32
        # b'=gc_minibatch=8
        # no_gc data = (2: input_ids, attention_mask), [b=32, n=2, e=16, l=32]
        # gc data = (2: input_ids, attention_mask), [b'=8, n=2, e=16, l=32]
        input_ids, attention_mask = data[0], data[1]

        B, N, E, _ = input_ids.shape

        input_ids = rearrange(input_ids, "b n e l -> (b n e) l")
        attention_mask = rearrange(attention_mask, "b n e l -> (b n e) l")

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )

        # at this point, we're embedding individual "comments"
        comment_embeddings = self.mean_pooling(
            outputs["last_hidden_state"], attention_mask
        )
        comment_embeddings = rearrange(
            comment_embeddings, "(b n e) l -> (b n) e l", b=B, n=N, e=E
        )

        # aggregate individual comments embeddings into episode embeddings
        episode_embeddings = self.attn_fn(
            comment_embeddings, comment_embeddings, comment_embeddings
        )
        episode_embeddings = reduce(episode_embeddings, "b e l -> b l", "max")

        episode_embeddings = self.linear(episode_embeddings)

        episode_embeddings = rearrange(episode_embeddings, "(b n) l -> b n l", b=B, n=N)
        # no_gc episode_embeddings = [bn=64, l=512], comment_embeddings = [bn=64, e=16, h=768]
        # gc episode_embeddings = [b'n=16, l=512], comment_embeddings = [b'n=16, e=16, h=768]
        return episode_embeddings, comment_embeddings

    def forward(self, *data):
        """Calculates a fixed-length feature vector for a batch of episode samples."""
        output = self.get_episode_embeddings(data)

        return output



if __name__ == "__main__":
    from transformers import AutoTokenizer

    checkpoint_path = "checkpoints/best-model.ckpt"
    state = torch.load(checkpoint_path, map_location="cpu")
    params = state["hyper_parameters"]['params']

    model = ERLAS(embedding_dim=params.embedding_dim, model_type=params.model_type, gradient_checkpointing=params.gradient_checkpointing)
    imcomplete_keys = model.load_state_dict(state["state_dict"], strict=False)
    print(f"imcomplete keys: {imcomplete_keys}")
    tokenizer = AutoTokenizer.from_pretrained(params.model_type)

    model.save_pretrained("erlas")
    tokenizer.save_pretrained("erlas")

    model.push_to_hub("Hieuman/erlas")
    tokenizer.push_to_hub("Hieuman/erlas")


