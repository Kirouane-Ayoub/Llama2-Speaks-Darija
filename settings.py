import torch

MAX_LENGTH = 512
VOCAB_SIZE = 1000
TRAIN_DATA_PATH = "train_eval_data/train_data.txt"
EVAL_DATA_PATH = "train_eval_data/eval_data.txt"
TOKENIZER_NAME = "hf-internal-testing/llama-tokenizer"


class ModelArgs:
    def __init__(
        self,
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=None,
        vocab_size=-1,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        mode="train",
        batch_size=32,
        max_seq_length=32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        pad_token_id=None,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.norm_eps = norm_eps
        self.mode = mode
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.device = device
        self.pad_token_id = pad_token_id


class TrainArgs(ModelArgs):
    def __init__(
        self, n_epochs=10, log_interval=12, lr=3e-4, warmup_steps=4000, **kwargs
    ):
        super().__init__(**kwargs)
        self.n_epochs = n_epochs
        self.log_interval = log_interval
        self.lr = lr
        self.warmup_steps = warmup_steps
