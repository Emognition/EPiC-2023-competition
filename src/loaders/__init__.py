from .epic import MULTI_EPIC


def get_dataset(dataset, dataset_dir, gtruth, sr=100, split="train", ecg_only=True):

    if dataset == "EPIC" and not ecg_only:
        return MULTI_EPIC(
            root=dataset_dir, sr=sr, scenario=2, split=split, category=gtruth, fold=0
        )
    else:
        raise NotImplementedError("Dataset not implemented")
