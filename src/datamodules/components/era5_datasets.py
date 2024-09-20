import pickle
from pathlib import Path

import numpy as np
import torch
# from torch_geometric.data import Data
from ffrecord import FileReader
from ffrecord.torch import Dataset, DataLoader
from torchvision.transforms import transforms

class ERA5(Dataset):
    def __init__(self,
                 data_dir: str,
                 split: str,
                 check_data: bool = True,
                 modelname: str = 'fourcastnet',
                 max_pred_step: int = 1) -> None:

        self.data_dir = Path(data_dir)

        assert split in ["train", "val", "test"]
        assert modelname in ["fourcastnet", "pangu"]
        self.split = split
        self.var_idx = np.load(f"{data_dir}/var_idx.npy")
        normalize_mean = np.load(f"{data_dir}/normalize_mean.npz")
        mean = []
        for var in self.var_idx:
            mean.append(normalize_mean[var])
        self.normalize_mean = np.reshape(np.concatenate(mean), (len(self.var_idx), 1, 1))
        normalize_std = np.load(f"{data_dir}/normalize_std.npz")
        self.normalize_std = np.reshape(np.concatenate([normalize_std[var] for var in self.var_idx]), (len(self.var_idx), 1, 1))
        self.modelname = modelname
        self.fname = str(self.data_dir / f"{split}.ffr")
        self.reader = FileReader(self.fname, check_data)
        self.max_pred_step = max_pred_step

    def __len__(self):
        return self.reader.n - self.max_pred_step

    def normalize(self, x):
        return (np.nan_to_num(x) - self.normalize_mean) / self.normalize_std

    def __getitem__(self, indices):
        # we read a batch of samples at once
        assert isinstance(indices, list)
        seqs_bytes = self.reader.read(indices)
        samples = []
        for bytes_ in seqs_bytes:
            xt0, xt1, y = pickle.loads(bytes_)
            samples.append((self.normalize(xt0), self.normalize(xt1), self.normalize(y)))

        return samples

    def loader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self, *args, **kwargs)