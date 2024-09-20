import pickle
from pathlib import Path

import numpy as np
import torch
# from torch_geometric.data import Data
from ffrecord import FileReader
from ffrecord.torch import Dataset, DataLoader


class StandardScaler:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def load(self, scaler_dir, var_idx_dir):
        with open(scaler_dir, "rb") as f:
            pkl = pickle.load(f)
            self.mean = pkl["mean"]
            self.std = pkl["std"]
        self.var_idx = np.load(var_idx_dir)
        # print(self.var_idx)

    def inverse_transform(self, data):
        mean, std = np.zeros(data.shape), np.ones(data.shape)
        # mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        # std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean


class Assim(Dataset):

    def __init__(self, data_dir: str, split: str, check_data: bool = True, modelname: str = 'fourcastnet') -> None:

        self.data_dir = Path(data_dir)

        assert split in ["train", "val", "test"]
        assert modelname in ["fourcastnet", "pangu"]
        self.split = split
        self.modelname = modelname
        self.fname = str(self.data_dir / f"{split}.ffr")
        self.reader = FileReader(self.fname, check_data)

    def __len__(self):
        return self.reader.n

    def __getitem__(self, indices):
        # we read a batch of samples at once
        assert isinstance(indices, list)
        seqs_bytes = self.reader.read(indices)
        samples = []
        for bytes_ in seqs_bytes:
            era5, xb, y_wind, y_t, y_r = pickle.loads(bytes_)
            # if self.modelname == 'fourcastnet':
            era5 = np.nan_to_num(era5)
            xb = np.nan_to_num(xb)
            y_wind = np.nan_to_num(y_wind)
            y_t = np.nan_to_num(y_t)
            y_r = np.nan_to_num(y_r)
            samples.append((era5, xb, y_wind, y_t, y_r))

        return samples

    def loader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self, *args, **kwargs)