import os
from torch.utils.data import Dataset
from dataset.zjumocap import ZJUMoCapDataset
from omegaconf import OmegaConf


class MultiPersonZJUMoCapDataset(Dataset):
    def __init__(self, cfg, split='train'):
        super().__init__()
        self.datasets = []
        self.person_ids = []

        # Essaye d'abord dans cfg (dataset level), sinon va chercher dans le parent
        dataset_names = getattr(cfg.dataset, "names", None)
        if dataset_names is None:
            dataset_names = getattr(cfg, "names", None)

        only_idx = getattr(cfg, "appearance_identity", None)
        mode = getattr(cfg, "mode", "train")

        if only_idx is not None and (
            split in ("val", "test", "predict") or 
            (split == "train" and mode in ("predict", "test"))
        ):
            try:
                dataset_names = [dataset_names[only_idx]]
            except Exception as e:
                raise ValueError(f"appearance_identity={only_idx} invalid for names={dataset_names}") from e
                
        if dataset_names is None:
            raise ValueError("Impossible de trouver `names` dans la config (ni dans cfg.dataset, ni dans cfg global)")

        print("Noms des sous-datasets =", dataset_names)
        self.identities = dataset_names
        self.split = split

        for pid, identity_name in enumerate(self.identities):
            if identity_name not in cfg.datasets:
                raise ValueError(f"{identity_name} non trouvé dans cfg.datasets")

            cfg_single = cfg.datasets[identity_name].dataset
            cfg_merged = OmegaConf.merge(cfg, OmegaConf.create({'dataset': cfg_single}))

            try:
                dataset = ZJUMoCapDataset(cfg_merged.dataset, split=split)
            except FileNotFoundError as e:
                if split == 'predict':
                    print(f"[predict] Skipping '{identity_name}' — {e}")
                    continue
                else:
                    raise

            if split == 'predict':
                train_ds = ZJUMoCapDataset(cfg_merged.dataset, split='train')
                dataset.metadata = train_ds.metadata
                print(f"[PREDICT] {identity_name}: metadata overridden with train metadata")
                print(f"[PREDICT] minimal_shape mean={train_ds.metadata['minimal_shape'].mean():.6f}")
            self.datasets.append(dataset)
            self.person_ids.append([pid] * len(dataset))

        self.cumulative_sizes = [0]
        for ds in self.datasets:
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + len(ds))
        self.all_person_ids = [pid for plist in self.person_ids for pid in plist]
        self.metadata = self.datasets[0].metadata

    def __len__(self):
        return self.cumulative_sizes[-1]

    def get_subjects(self):
        return self.identities

    def __getitem__(self, index):
        dataset_idx = next(i for i, cs in enumerate(self.cumulative_sizes) if index < cs) - 1
        sample_idx = index - self.cumulative_sizes[dataset_idx]
        sample = self.datasets[dataset_idx][sample_idx]

        if hasattr(sample, 'data'):
            sample.data['person_id'] = dataset_idx
            setattr(sample, 'person_id', dataset_idx)
        else:
            sample['person_id'] = dataset_idx

        return sample