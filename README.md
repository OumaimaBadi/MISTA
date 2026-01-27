# MISTA

MISTA addresses the challenge of learning a shared and compact representation of multi-identity human avatars from monocular videos. Most existing approaches rely on per-identity optimization, which is costly in both time and storage and limits generalization.

We propose MISTA, a method that applies Tensor Train (TT) decomposition to approximate a global tensor of 3D Gaussian (3DG) parameters for multiple avatars in a canonical pose. Before decomposition, parameters are reordered and reshaped in a structure-aware manner to better exploit spatial correlations.

Experiments show that MISTA achieves reconstruction and animation quality comparable to MIGS while using only 6.5% of its parameters, making it suitable for deployment on resource-constrained devices.

## Code

Our code will be available soon.

## Citation

If you find our work helpful, please consider citing:

```bibtex
@inproceedings{badi2025mista,
  title     = {MISTA: Compact Multi-Identity Structure-Aware Tensorized Avatars},
  author    = {Oumaima Oumaima and Xiaoran Jiang and Luce Morin and Marten Sjostrom},
  booktitle = {Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2026}
}


