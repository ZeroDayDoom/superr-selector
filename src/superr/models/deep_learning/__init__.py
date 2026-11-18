"""Deep learning: multi-task MLP and GRU sequence model (PyTorch).

The MLP follows the specified architecture:
  input -> numeric normalisation + categorical embeddings
        -> Dense 256 -> BatchNorm -> ReLU -> Dropout
        -> Dense 128 -> ReLU -> Dropout
        -> Dense 64 -> ReLU
        -> heads (outcome distribution CE + event BCE heads)

Deep models are EXPERIMENTS: they are selected only if they beat the
tabular models on the temporal holdout (Phase-6 Part G).
"""
