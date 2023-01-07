from pathlib import Path
from typing import Union
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class EpilepticDataset(Dataset):

    def __init__(self, root_data_dir: Union[str, Path], metadata: pd.DataFrame, transforms=None):
        self.root_data_dir = Path(root_data_dir)
        self.metadata_df = metadata
        self.transforms = transforms
        self.balance = True
        self.data = self.load_data()

    def load_data(self):
        data = {}
        # for debugging purposes only load 2 files
        for file in os.listdir(self.root_data_dir / "windows_data"):
            data[file] = torch.tensor(np.load(self.root_data_dir / "windows_data" / file)["arr_0"])
            print(file, data[file].shape)
        return data

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        metadata = self.metadata_df.iloc[idx]
        filename = "chb" + str(metadata["pacient"]).zfill(2) + "_raw_eeg_128.npz"
        print(f"\nDEBUG: {self.data.keys()}\n")
        signal = self.data[filename][metadata["index"]]
        # Permute (w, c) -> (c, w)
        signal = signal.permute((1, 0))

        target = torch.tensor(metadata["label"], dtype=torch.long)

        metadata = metadata[["id", "pacient", "index_inicial", "periode", "recording"]].to_dict()

        sample = {"signal": signal, "target": target, "metadata": metadata}

        # if self.balance:
        #     # if class is 1, randomly discard 90% of the samples
        #     if target == 1:
        #         if np.random.rand() < 0.9:
        #             return None

        if self.transforms:
            sample["signal"] = self.transforms(sample["signal"])

        return sample
