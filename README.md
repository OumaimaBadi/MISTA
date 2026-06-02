# MISTA: Compact Multi-Identity Structure-Aware Tensorized Avatars

## Overview

This repository contains the official implementation of **MISTA** and **MISTA-AR**, two tensorized Gaussian avatar representations designed for efficient multi-identity human modeling, animation, and motion transfer.

MISTA introduces a structure-aware Tensor Train factorization of 3D Gaussian Splatting parameters, enabling compact multi-identity representations while preserving rendering quality.

MISTA-AR extends MISTA through adaptive rank selection using MARS, allowing automatic identification and pruning of less important tensor components during training.

The framework supports:

* Multi-identity avatar modeling
* Motion transfer between subjects
* Novel pose synthesis
* Tensorized Gaussian representations
* Adaptive rank selection
* Tensor Train and CP-based representations
* Efficient compression of Gaussian avatar parameters

---

## Associated Publications

This repository accompanies the following publications:

### MISTA

O. Badi, X. Jiang, L. Morin, and M. Sjöström,

**"MISTA: Compact Multi-Identity Structure-Aware Tensorized Avatars"**

Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026.

### MISTA-AR

O. Badi, X. Jiang, L. Morin, and M. Sjöström,

**"Une représentation 3DGS de rangs faibles auto-sélectionnés pour les avatars multi-identités"**

Actes de CORESA 2026, Nantes, France, 2026.

---

## Features

* Multi-identity Gaussian avatar modeling
* Tensor Train tensorization
* CP decomposition baseline
* Adaptive rank selection with MARS
* Motion transfer
* Novel-view synthesis
* Novel-pose synthesis
* ZJU-MoCap support
* Singularity-based deployment

---

## Installation

### Clone Repository

```bash
git clone https://github.com/OumaimaBadi/MISTA.git
cd MISTA
```

---

## Environment

A ready-to-use Singularity image is provided.

Download links:

| Resource                   | Link        |
| -------------------------- | ----------- |
| Singularity image          | Coming soon |
| MISTA pretrained models    | Coming soon |
| MISTA-AR pretrained models | Coming soon |
| MIGS Rank-10 checkpoints   | Coming soon |
| MIGS Rank-100 checkpoints  | Coming soon |

---

## SMPL Setup

Download SMPL models from the official SMPL website and place them under:

```text
body_models/
└── smpl/
    ├── male/
    ├── female/
    └── neutral/
```

Then run:

```bash
python extract_smpl_parameters.py
```

---

## Dataset Preparation

Due to licensing restrictions, preprocessed datasets cannot be redistributed.

The repository currently supports:

* ZJU-MoCap
* SMPL-based custom datasets

Please prepare datasets according to the configuration files located in:

```text
configs/dataset/
```

---

## Repository Structure

```text
configs/
dataset/
gaussian_renderer/
human_body_prior/
models/
preprocess_datasets/
scene/
submodules/
utils/

environment.yml
extract_smpl_parameters.py
prune_model.py
render.py
train5d_mars.py
```

---

## Training

Before training, modify the following variables inside the provided SLURM script:

```bash
SIF=/path/to/your/singularity_image.sif

BIND_DATA=/path/to/your/dataset:/data/

BIND_SRC=/path/to/your/source_code:/src/

WANDB_API_KEY=<your_wandb_key>
```

---

### MISTA Training

```bash
python train5d_mars.py \
dataset=migs_multi_zju_5d \
migs.type=tt5d \
migs.use_mars=false
```

---

### MISTA-AR Training

```bash
python train5d_mars.py \
dataset=migs_multi_zju_5d_mars \
migs.type=tt5d \
migs.use_mars=true
```

---

### MIGS (CP) Training

```bash
python train5d_mars.py \
migs.type=cp
```

---

## Rendering

### Evaluation

```bash
python render.py mode=test
```

### Novel View Synthesis

```bash
python render.py \
mode=test \
dataset.test_mode=view
```

### Novel Pose Synthesis

```bash
python render.py \
mode=test \
dataset.test_mode=pose
```

---

## Motion Transfer

Motion transfer can be performed by applying the motion sequence of a source identity to a target identity while preserving the target appearance.

Example:

```bash
python render.py mode=predict
```

The target identity is reconstructed using its learned appearance and animated using the source motion sequence.

---

## Pretrained Models

The following checkpoints are provided:

| Model     | Description                                     |
| --------- | ----------------------------------------------- |
| MIGS-R10  | CP decomposition with rank 10                   |
| MIGS-R100 | CP decomposition with rank 100                  |
| MISTA     | Tensor Train representation                     |
| MISTA-AR  | Tensor Train representation with adaptive ranks |

Download links will be added upon release.

---

## Citation

If you use this repository, pretrained models, datasets, or any part of this work in your research, please cite:

```bibtex
@inproceedings{badi2026mista,
  title={MISTA: Compact Multi-Identity Structure-Aware Tensorized Avatars},
  author={Badi, Oumaima and Jiang, Xudong and Morin, Luce and Sjöström, Mårten},
  booktitle={Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}

@inproceedings{badi2026mista_ar,
  title={Une représentation 3DGS de rangs faibles auto-sélectionnés pour les avatars multi-identités},
  author={Badi, Oumaima and Jiang, Xudong and Morin, Luce and Sjöström, Mårten},
  booktitle={Actes de CORESA},
  address={Nantes, France},
  year={2026}
}
```

---

## License

This repository is intended for research and academic purposes.

Please refer to the LICENSE file for additional details.

---

## Acknowledgements

This project builds upon ideas, datasets, and open-source implementations from:

* 3DGS-Avatar
* 3D Gaussian Splatting
* MIGS
* 4D-Humans
* AIST++
* ZJU-MoCap

We sincerely thank the authors of these works for making their research and resources publicly available.
