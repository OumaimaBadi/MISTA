from .zjumocap import ZJUMoCapDataset
from .people_snapshot import PeopleSnapshotDataset
from .MultiPersonZJUMoCap import MultiPersonZJUMoCapDataset
from .neuman import NeuManDataset
from .multi_person_neuman import MultiPersonNeuManDataset

def load_dataset(cfg, split='train'):
    dataset_dict = {
        'zjumocap': ZJUMoCapDataset,
        'MultiPersonZJUMoCap': MultiPersonZJUMoCapDataset,
        'people_snapshot': PeopleSnapshotDataset,
        'neuman': NeuManDataset,
        'MultiPersonNeuMan': MultiPersonNeuManDataset,
    }

    dataset_name = cfg.dataset.name
    dataset_class = dataset_dict[dataset_name]

    if dataset_name in ["MultiPersonZJUMoCap", "MultiPersonNeuMan"]:
        return dataset_class(cfg, split=split)
    else:
        return dataset_class(cfg.dataset, split=split)
