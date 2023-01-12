"""
This script will test the results from the trained models
1. import the checkpoint
2. load the data from the test set in a datamodule object
3. evaluate the model with the ordered test set
4. calculate metrics (accuracy, precision, recall, f1, confusion matrix)
5. plot the confusion matrix
6. plot some true positives, false positives, true negatives and false negatives
"""

import os
import sys

from tqdm import tqdm
import seaborn as sns

#from src.deep_learning.data.datamodule import DataModule
from src.deep_learning.transforms.common import ZScoreNormalize, L2Normalize
from torchvision import transforms
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from tsai.models.RNNPlus import GRUPlus, LSTMPlus, RNNPlus
from tsai.models.FCNPlus import FCNPlus
from src.deep_learning.models.lightning_module import LightningModule


import torch

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

class custom_testing_dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.len = len(self.data)
        self.transforms = transforms.Compose([ZScoreNormalize(), L2Normalize()])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        sample = self.data[idx]
        label = self.labels[idx]
        sample_tensor = torch.tensor(sample, dtype=torch.float32)
        # permute
        sample_tensor = sample_tensor.permute((1, 0))

        label_tensor = torch.tensor(label, dtype=torch.long)

        sample_transformed = self.transforms(sample_tensor)

        return sample_transformed, label_tensor

        

if __name__=="__main__":
    # Import the checkpoint
    checkpoint_path = "checkpoints/best_model/model_epoch=15_val_loss=0.36.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    #model.load_state_dict(checkpoint["state_dict"])
    model = LightningModule.load_from_checkpoint(checkpoint_path)
    root_data_dir = Path("data/").resolve()


    # Load the data from the test set in a datamodule object
    test_data_path = "data/testing_data/chb01_raw_eeg_128.npz"
    test_metadata_path = "data/testing_data/chb01_raw_eeg_128.csv"
    # Load the npz
    test_data = np.load(test_data_path)["arr_0"]
    test_metadata = pd.read_csv(test_metadata_path)
    
    # Create a custom dataset
    test_dataset = custom_testing_dataset(test_data, test_metadata["label"])

    # Create a dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Evaluate the model with the ordered test set
    model.eval()
    model.freeze()

    preds = []
    results = {
        "true_1": {
            "num": 0,
            "data": []
        },
        "false_1": {
            "num": 0,
            "data": []
        },
        "true_0": {
            "num": 0,
            "data": []
        },
        "false_0": {
            "num": 0,
            "data": []
        }
    }

    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Testing"):
            x, y = data
            y_hat = model(x)
            # print(y_hat[0].argmax().item())
            # print(y.item())
            preds.append(y_hat[0].argmax().item())
            # Reverse the permute and the transforms
            original_data = x[0].permute((1, 0))
            original_data = original_data.numpy()

            if y_hat[0].argmax().item() == 1 and y.item() == 1:
                results["true_1"]["num"] += 1
                results["true_1"]["data"].append(x)
            elif y_hat[0].argmax().item() == 1 and y.item() == 0:
                results["false_1"]["num"] += 1
                results["false_1"]["data"].append(x)
            elif y_hat[0].argmax().item() == 0 and y.item() == 0:
                results["true_0"]["num"] += 1
                results["true_0"]["data"].append(x)
            elif y_hat[0].argmax().item() == 0 and y.item() == 1:
                results["false_0"]["num"] += 1
                results["false_0"]["data"].append(x)

    
    print("True 1: ", results["true_1"]["num"])
    print("False 1: ", results["false_1"]["num"])
    print("True 0: ", results["true_0"]["num"])
    print("False 0: ", results["false_0"]["num"])

    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns



    original = test_metadata["label"]

    cm = confusion_matrix(original, preds)
    print(cm)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.savefig("confusion_matrix.png")

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    print("Accuracy: ", accuracy_score(original, preds))
    print("Precision: ", precision_score(original, preds))
    print("Recall: ", recall_score(original, preds))
    print("F1: ", f1_score(original, preds))

    # calculate accuracy per each class
    from sklearn.metrics import classification_report
    print(classification_report(original, preds))

    # Create the original timelane of labels as a horizonal bar, to compare with the predictions
    # Create a list of 0s and 1s
    labels = original.tolist()
    preds = preds
    # Plot the labels and preds in a 2x1 grid as vertical bars, colored with the labels
    fig, axs = plt.subplots(2,1 , figsize=(20, 10))
    original_colors = [(1, 0, 0) if label == 1 else (0, 0, 1) for label in labels]
    axs[0].bar(range(len(labels)), color=original_colors, width=1)
    axs[0].set_title("Labels")
    preds_colors = [(1, 0, 0) if pred == 1 else (0, 0, 1) for pred in preds]
    axs[1].bar(range(len(preds)), color=preds_colors, width=1)
    axs[1].set_title("Predictions")


    plt.savefig("labels_preds.png")

    # Display some true positives, false positives, true negatives and false negatives
    # Create a 5x4 grid, with each group on a row
    fig, axs = plt.subplots(5, 4, figsize=(20, 20))
    for i in range(5):
        axs[i, 0].imshow(results["true_1"]["data"][i][0].permute((1, 0)).numpy())
        axs[i, 0].set_title("True 1")
        axs[i, 1].imshow(results["false_1"]["data"][i][0].permute((1, 0)).numpy())
        axs[i, 1].set_title("False 1")
        axs[i, 2].imshow(results["true_0"]["data"][i][0].permute((1, 0)).numpy())
        axs[i, 2].set_title("True 0")
        axs[i, 3].imshow(results["false_0"]["data"][i][0].permute((1, 0)).numpy())
        axs[i, 3].set_title("False 0")
    plt.savefig("true_false.png")













            
