from pathlib import Path
from typing import Dict, Union

import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.deep_learning.data.dataset import EpilepticDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, root_data_dir: Union[str, Path] = './data', batch_size: int = 32, num_workers: int = 4,
                 transforms: Dict = {"train": None, "valid": None, "test": None}):
        super().__init__()
        self.root_data_dir = Path(root_data_dir).resolve()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = EpilepticDataset
        self.transforms = transforms

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            full_train_metadata_df = pd.read_csv(self.root_data_dir / "metadata" / "chb01_raw_eeg_128_full.csv")
            train_metadata_df, valid_metadata_df = train_test_split(full_train_metadata_df, test_size=0.2,
                                                                    random_state=0,
                                                                    stratify=full_train_metadata_df['label'])

            self.train_ds = self.dataset(self.root_data_dir, train_metadata_df, transforms=self.transforms["train"])
            self.valid_ds = self.dataset(self.root_data_dir, valid_metadata_df, transforms=self.transforms["valid"])

        if stage == 'test' or stage is None:
            # TODO test csv
            test_metadata_df = pd.read_csv(self.root_data_dir / "metadata" / "chb01_raw_eeg_128_full.csv")
            self.test_ds = self.dataset(self.root_data_dir, test_metadata_df, transforms=self.transforms["test"])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)
