from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
#from tsai.all import *

#from tsai.models.TransformerPlus import TransformerPlus
from tsai.models.FCNPlus import FCNPlus
from tsai.models.ResNetPlus import ResNetPlus
from tsai.models.XceptionTimePlus import XceptionTimePlus
from tsai.models.RNNPlus import GRUPlus, LSTMPlus, RNNPlus
from tsai.models.TSSequencerPlus import TSSequencerPlus
from tsai.models.XResNet1dPlus import xresnet1d50_deeperplus
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tsai.models.RNN_FCNPlus import MGRU_FCNPlus, MLSTM_FCNPlus, MRNN_FCNPlus
from tsai.models.TSTPlus import TSTPlus

from src.deep_learning.data.datamodule import DataModule
from src.deep_learning.models.lightning_module import LightningModule
from src.deep_learning.transforms.common import ZScoreNormalize, L2Normalize

import optuna
class HyperparameterOptimization:

    def __init__(self, root_data_dir, data_module):
        self.root_data_dir = root_data_dir
        self.data_module = data_module

    def objective(self,trial):
        """
        Objective function to be optimized.
        """
        # Choose model

        MODEL_NAME = trial.suggest_categorical("model", ["FCNPlus", "ResNetPlus", "XceptionTimePlus", "LSTMPlus", "RNNPlus", "TSSequencerPlus", "InceptionTimePlus", "TSTPlus"])
        print("Model:", MODEL_NAME)
        # MODEL_NAME = "xresnet1d50_deeperplus"
        if MODEL_NAME == "FCNPlus":
            model = FCNPlus(21, 2)
        elif MODEL_NAME == "ResNetPlus":
            model = ResNetPlus(21, 2)
        elif MODEL_NAME == "XceptionTimePlus":
            model = XceptionTimePlus(21, 2)
        elif MODEL_NAME == "GRUPlus":
            model = GRUPlus(21, 2)
        elif MODEL_NAME == "LSTMPlus":
            model = LSTMPlus(21, 2)
        elif MODEL_NAME == "RNNPlus":
            model = RNNPlus(21, 2)
        elif MODEL_NAME == "TSSequencerPlus":
            model = TSSequencerPlus(21, 2, 128)
        elif MODEL_NAME == "xresnet1d50_deeperplus":
            model = xresnet1d50_deeperplus(21, 2)
        elif MODEL_NAME == "InceptionTimePlus":
            model = InceptionTimePlus(21, 2)
        elif MODEL_NAME == "MGRU_FCNPlus":
            model = MGRU_FCNPlus(21, 2, 128)
        elif MODEL_NAME == "MLSTM_FCNPlus":
            model = MLSTM_FCNPlus(21, 2, 128)
        elif MODEL_NAME == "MRNN_FCNPlus":
            model = MRNN_FCNPlus(21, 2, 128)
        elif MODEL_NAME == "TSTPlus":
            model = TSTPlus(21, 2, 128)
        else:
            raise ValueError("Invalid model name.")

        # Create LightningModule
        num_classes = 2
        LEARNING_RATE = 1e-2 #trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
        model = LightningModule(model, num_classes=num_classes, learning_rate=LEARNING_RATE)

        # Logger
        wandb_logger = WandbLogger(project='epileptic-detection', job_type='train')

        # Callbacks
        callbacks = [
            # EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="max"),
            LearningRateMonitor(),
            ModelCheckpoint(dirpath="./checkpoints", monitor="val_loss", filename="model_{MODEL_NAME}_{epoch:02d}_{val_loss:.2f}")
        ]

        # Create trainer
        trainer = pl.Trainer(max_epochs = 10,
                             check_val_every_n_epoch=None,
                             # gpus=0,
                             # devices=[0],
                             logger=wandb_logger,
                             callbacks=callbacks,
                             enable_progress_bar=True
                             # accelerator="gpu"
                             )


        trainer.fit(model, self.data_module)

        return model.min_loss
    
