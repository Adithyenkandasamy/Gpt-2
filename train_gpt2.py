from dataclasses import dataclass
import torch 
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 256  # Maximum sequence length
    vocab_size: int = 65   # Vocabulary size (number of unique tokens)
    n_layer: int = 6       # Number of transformer blocks (layers)
    n_head: int = 6        # Number of attention heads
    n_embd: int = 384      # Embedding dimension (size of token representation)

class GPT(nn.Model):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModelDict(dict{
            wte = nn.Embedding(config.vocab_size, confign.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)