from pathlib import Path

import pytest
import torch
import sys
sys.path.append('.')
from src.datamodules.weatherbench_datamodule import WEATHERBENCHDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_weatherbench_datamodule(batch_size):
    data_dir = "/public/home/wangwuxing01/research/weatherbench/data/"
    dm = WEATHERBENCHDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "data.ffr").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    print(num_datapoints)

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float64
    assert y.dtype == torch.float64

if __name__ == '__main__':
    pytest.main('-s', 'test_weatherbench_datamodule.py')