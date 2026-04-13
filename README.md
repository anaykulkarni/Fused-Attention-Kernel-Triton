# Fused Attention Kernels with Triton
A hands-on tutorial exploring positional embeddings in Transformer attention, culminating in a fused [Triton](https://triton-lang.org/) kernel that combines RoPE, scaled dot-product attention, and online softmax into a single GPU pass.

## Overview

Transformers are permutation-invariant — without positional information, self-attention can't distinguish token order. This tutorial covers two approaches to fixing that:

1. **Sinusoidal Positional Encoding** ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) — adds a fixed, frequency-based vector to each token embedding.
2. **Rotary Position Embeddings (RoPE)** ([Su et al., 2021](https://arxiv.org/abs/2104.09864)) — rotates query and key vectors in 2D subspaces so the dot product naturally encodes relative position. Used in LLaMA, Mistral, and most modern LLMs.

We then benchmark three implementations of multi-head attention with RoPE:

| Implementation | Description |
|---|---|
| Naive PyTorch | Loop over heads, full attention matrix materialized in HBM |
| `torch.compile` | Same code with automatic kernel fusion |
| **Triton Fused Kernel** | Hand-written kernel fusing RoPE + attention + online softmax |

## Results

| Implementation | Time (100 iters) | Speedup |
|---|---|---|
| Naive PyTorch | ~5.36 s | 1.00x |
| `torch.compile` | ~2.66 s | ~2.01x |
| **Triton Fused** | **~1.96 s** | **~2.73x** |

## Notebooks

| Notebook | Contents |
|---|---|
| [`fused-attention-with-rope.ipynb`](fused-attention-with-rope.ipynb) | Full tutorial — sinusoidal encoding, RoPE, and all three benchmark implementations |
| [`fused-attention.ipynb`](fused-attention.ipynb) | Earlier version without RoPE |

## Key Ideas

- **Sinusoidal vs RoPE**: Sinusoidal encoding is additive and parameter-free but lives in the residual stream where it can be diluted. RoPE embeds position directly in the attention scores via rotation, making it more robust.

- **Interleaved RoPE in Triton**: The kernel loads even (`q[..., 0::2]`) and odd (`q[..., 1::2]`) columns separately via gather pointers, applies the 2D rotation, then exploits the identity Q_rot @ K_rot^T = Q_a' @ K_a'^T + Q_b' @ K_b'^T to avoid reconstructing the interleaved layout in registers.

- **Online softmax** ([Milakov & Gimelshein, 2018](https://arxiv.org/abs/1805.02867)): Enables single-pass attention by maintaining running max and sum statistics, avoiding O(N²) memory for the full attention matrix. This is the core idea behind [FlashAttention](https://arxiv.org/abs/2205.14135).

## Setup

```bash
pip install torch triton
```

Tested with Python 3.12, PyTorch 2.x, Triton. Requires an NVIDIA GPU.

### Configuration

The tutorial uses:
- Sequence length: 2048
- Hidden dimension: 4096
- Attention heads: 128 (head dim = 32)
- Precision: bfloat16
