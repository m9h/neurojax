"""Electrophysiology tokenization for JAX/Equinox.

Ports EphysTokenizer (OHBA, Oxford) to JAX and extends it with
VQ-VAE/FSQ quantization and neurojax-native extension tokenizers.
"""

from neurojax.tokenizer._tokenizer import EphysTokenizer, TokenizerOutput
from neurojax.tokenizer._encoder import RNNEncoder
from neurojax.tokenizer._quantizer import (
    TemperatureQuantizer,
    VQQuantizer,
    FSQQuantizer,
)
from neurojax.tokenizer._decoder import Conv1dDecoder
from neurojax.tokenizer._baselines import (
    mu_law_tokenize,
    mu_law_detokenize,
    quantile_tokenize,
)
from neurojax.tokenizer._metrics import pve, pve_per_channel, token_utilization
from neurojax.tokenizer._vocab import refactor_vocabulary
from neurojax.tokenizer._train import fit
from neurojax.tokenizer._extensions import (
    SignatureTokenizer,
    RiemannianTokenizer,
    TFRTokenizer,
)
from neurojax.tokenizer._dynamics_reg import KoopmanRegularizer
from neurojax.tokenizer._consumer import TokenConsumer, evaluate_tokenizer

__all__ = [
    "EphysTokenizer",
    "TokenizerOutput",
    "RNNEncoder",
    "TemperatureQuantizer",
    "VQQuantizer",
    "FSQQuantizer",
    "Conv1dDecoder",
    "mu_law_tokenize",
    "mu_law_detokenize",
    "quantile_tokenize",
    "pve",
    "pve_per_channel",
    "token_utilization",
    "refactor_vocabulary",
    "fit",
    "SignatureTokenizer",
    "RiemannianTokenizer",
    "TFRTokenizer",
    "KoopmanRegularizer",
    "TokenConsumer",
    "evaluate_tokenizer",
]
