"""
TurboQuant Inference Engine (TQE)

A production-ready implementation of TurboQuant (ICLR 2026, Google Research):
  - Online, data-oblivious vector quantization with near-optimal distortion
  - Drop-in KV cache compressor for HuggingFace LLMs
  - Compressed approximate nearest-neighbor search engine

References:
  TurboQuant: arXiv:2504.19874
  QJL:        arXiv:2406.03482
  PolarQuant: arXiv:2502.02617
"""

__version__ = "0.1.0"
__author__ = "TurboQuant Inference Engine Contributors"

from tqe.algorithms.qjl import QJLQuantizer
from tqe.algorithms.polar_quant import PolarQuantizer
from tqe.algorithms.turbo_quant import TurboQuantizer

__all__ = [
    "QJLQuantizer",
    "PolarQuantizer",
    "TurboQuantizer",
]
