from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from tsai.models.FCNPlus import FCNPlus

from src.deep_learning.data.datamodule import DataModule
from src.deep_learning.models.lightning_module import LightningModule

def main():
    root_data_dir = Path("../data/").resolve()

    # Create datamodule
    dm = DataModule(root_data_dir, batch_size=4)
    dm.setup()

    # Choose model
    tsai_model = FCNPlus(21, 2)

    # Create LightningModule
    num_classes = 2
    model = LightningModule(tsai_model, num_classes=num_classes, learning_rate=1e-3)

    # Logger
    wandb_logger = WandbLogger(project='epileptic-detection', job_type='train')

    # Callbacks
    callbacks = [
        # EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="max"),
        # LearningRateMonitor(),
        ModelCheckpoint(dirpath="./checkpoints", monitor="val_loss", filename="model_{epoch:02d}_{val_loss:.2f}")
    ]

    # Create trainer
    trainer = pl.Trainer(max_steps=1000,
                         val_check_interval=500,
                         gpus=0,
                         logger=wandb_logger,
                         callbacks=callbacks,
                         enable_progress_bar=True)

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
