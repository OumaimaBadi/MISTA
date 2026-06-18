# MISTA: Compact Multi-Identity Structure-Aware Tensorized Avatars

## Overview

This repository contains the official implementation of **MISTA** and **MISTA-AR**, two tensorized Gaussian avatar representations designed for efficient multi-identity human modeling, animation, and motion transfer.

MISTA introduces a structure-aware Tensor Train factorization of 3D Gaussian Splatting parameters, enabling compact multi-identity representations while preserving rendering quality.

MISTA-AR extends MISTA through adaptive rank selection using MARS, allowing automatic identification and pruning of less important tensor components during training.

The framework supports:

* Multi-identity avatar modeling
* Motion transfer between subjects
* Novel pose synthesis
* Adaptive rank selection
* Tensor Train and CP-based representations

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
| Singularity image          | [Download](https://drive.google.com/file/d/1qlvbzLRj3HTjnJF7AG4D-blOPADvj9g-/view?usp=drive_link) |
| MISTA pretrained model    | [Download](https://drive.google.com/file/d/1j2c7ZdcAfkJSGyjlgxKfHi7q6fhVR3m1/view?usp=drive_link) |
| MISTA no Hilbert pretrained model    |[Download](https://drive.google.com/file/d/1MQIz64GXHD_JzIzi8bGdXZkqZeNF8NxS/view?usp=drive_link) |
| MISTA-AR pretrained model | [Download](https://drive.google.com/file/d/1fgibV4yhadyJQmyRdX6qZKsL_t8yD8eR/view?usp=drive_link) |
| MIGS Rank-10 checkpoint   | [Download](https://drive.google.com/file/d/1bCRP93p9esL6pheGFOYEogQhkRoLnDLQ/view?usp=drive_link) |
| MIGS Rank-100 checkpoint  | [Download](https://drive.google.com/file/d/1FtSojegVHJCrx5OJbCoxA_uqyVL-y9Xc/view?usp=drive_link) |

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

Due to dataset licensing restrictions, we cannot publicly distribute the preprocessed ZJU-MoCap data.

Users should download the original dataset and follow the preprocessing procedure described in the ARAH repository:

https://github.com/taconite/arah-release

The resulting processed data can then be used directly with the MISTA framework.

Please prepare datasets according to the configuration files located in:

```text
configs/dataset/
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
dataset.test_mode=view \
dataset=migs_multi_zju_5d_mars \
opt.iterations=50000 \
migs.type=tt5d \
migs.use_mars=false \
appearance_identity=0 // 0:386, 1:387, 2:377, 3:392, 4:315, 5:394, 6:393, 7:390
load_ckpt=/path/to/ckpt50000.pth
```

### Novel Pose Synthesis

```bash
python render.py mode=predict \
dataset=migs_multi_zju_5d_mars \
opt.iterations=50000 \
migs.type=tt5d \
migs.use_mars=false \
dataset.predict_seq= 0 // 0,1,2,3, to try differnt dances
appearance_identity=0 // 0:386, 1:387, 2:377, 3:392, 4:315, 5:394, 6:393, 7:390
load_ckpt=/path/to/ckpt50000.pth
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
  author={Badi, Oumaima and Jiang, Xiaoran and Morin, Luce and Sjöström, Mårten},
  booktitle={Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}

@inproceedings{badi2026mista_ar,
  title={Une représentation 3DGS de rangs faibles auto-sélectionnés pour les avatars multi-identités},
  author={Badi, Oumaima and Jiang, Xiaoran and Morin, Luce and Sjöström, Mårten},
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
