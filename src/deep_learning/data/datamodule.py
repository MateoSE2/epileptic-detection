from pathlib import Path
from typing import Dict, Union
import os

import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.deep_learning.data.dataset import EpilepticDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, root_data_dir: Union[str, Path] = './data', batch_size: int = 32, num_workers: int = 4,
                 transforms: Dict = {"train": None, "valid": None, "test": None}, balanced: bool = False):
        super().__init__()
        self.root_data_dir = Path(root_data_dir).resolve()
        self.batch_size = batch_size
        self.num_workers = num_workers
        #self.num_workers = 0
        self.balanced = balanced

        self.dataset = EpilepticDataset
        self.transforms = transforms

    def setup(self, stage: str = None):
        if self.balanced:
            print("Balanced dataset")
            metadata_filename = "balanced_metadata"
        else:
            print("Unbalanced dataset")
            metadata_filename = "metadata"
        if stage == 'fit' or stage is None:
            df_lists = []
            full_train_metadata_df = pd.DataFrame()
            for file in os.listdir(self.root_data_dir / metadata_filename):
                df_tmp = pd.read_csv(self.root_data_dir / metadata_filename / file)
                df_tmp = df_tmp.reset_index()
                df_lists.append(df_tmp)

            full_train_metadata_df = pd.concat(df_lists, ignore_index=True)

            print(f"Pre-split (before discarding):\n{full_train_metadata_df.label.value_counts()}")

            # Discard % of the samples with label 0
            if not self.balanced:
                percentage_to_discard = 0.9
                full_train_metadata_df = full_train_metadata_df[full_train_metadata_df.label == 1]
                full_train_metadata_df = full_train_metadata_df.append(df_lists[0][df_lists[0].label == 0].sample(frac=1-percentage_to_discard))

                print(f"Pre-split (after discarding {percentage_to_discard*100}% of label 1):\n{full_train_metadata_df.label.value_counts()}")


            train_metadata_df, valid_metadata_df = train_test_split(full_train_metadata_df, test_size=0.2,
                                                                    random_state=0,
                                                                    stratify=full_train_metadata_df['label'])

            print(f"Train:\n{train_metadata_df.label.value_counts()}")
            print(f"Valid:\n{valid_metadata_df.label.value_counts()}")

            self.train_ds = self.dataset(self.root_data_dir, train_metadata_df, transforms=self.transforms["train"], balanced=self.balanced)
            self.valid_ds = self.dataset(self.root_data_dir, valid_metadata_df, transforms=self.transforms["valid"], balanced=self.balanced)

        if stage == 'test' or stage is None:
            test_metadata_df = pd.DataFrame()
            for file in os.listdir(self.root_data_dir / "metadata"):
                test_metadata_df = test_metadata_df.append(pd.read_csv(self.root_data_dir / "metadata" / file), ignore_index=True)

            self.test_ds = self.dataset(self.root_data_dir, test_metadata_df, transforms=self.transforms["test"])

    def train_dataloader(self):
       return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)
