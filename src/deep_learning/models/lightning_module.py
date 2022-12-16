import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics


class LightningModule(pl.LightningModule):
    """Expects a tsai model as an input with a backbone and a head."""

    def __init__(self, model, num_classes=2, learning_rate=1e-2):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.num_classes = num_classes
        self.model = model

        self.softmax = nn.Softmax()
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(num_classes=num_classes)
        self.f_score = torchmetrics.F1Score(task='binary')

    def forward(self, x, metadata):
        x = self.model.backbone(x)
        x = self.model.head(x)
        x = self.softmax(x)
        return x

    def training_step(self, batch, batch_idx):
        images, target, metadata = batch["signal"], batch["target"], batch["metadata"]
        logits = self.forward(images, metadata)
        loss = self.loss(logits, target)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, target.int())
        f_score = self.f_score(preds, target.int())
        self.log('train_loss', loss, on_step=True, logger=True)
        self.log('train_acc', acc, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, target, metadata = batch["signal"], batch["target"], batch["metadata"]
        logits = self.forward(images, metadata)
        loss = self.loss(logits, target)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, target.int())
        f_score = self.f_score(preds, target.int())
        self.log('val_loss', loss, on_step=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, target, metadata = batch["signal"], batch["target"], batch["metadata"]
        logits = self.forward(images, metadata)
        loss = self.loss(logits, target)

        # test metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, target)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}