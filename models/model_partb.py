import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
import numpy as np
from torch.utils.data import DataLoader,Dataset
from torchvision.models import ResNet50_Weights
import torchvision

class FinetuneImgModel(pl.LightningModule):
    def __init__(self, train_dataset, test_dataset, val_dataset, lr=3e-4, num_classes=10, batch_size=16):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.lr = lr
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.best_val_acc = 0

        pretrained_model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Freeze all the layers except the last fully connected layer
        for param in pretrained_model.parameters():
            param.requires_grad = False
        
        self.fc1 = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.output = nn.Linear(pretrained_model.fc.in_features, self.num_classes)
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        metrics = {"loss": loss, "train_acc": acc}
        self.log_dict(metrics, on_epoch=True)
        return metrics
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        if acc >= self.best_val_acc:
            self.best_val_acc = acc
        metrics = {"val_loss": loss, "val_acc": acc, "best_val_acc": self.best_val_acc}
        self.log_dict(metrics)
        return metrics
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        metrics = {"test_loss": loss, "test_acc": acc}
        self.log_dict(metrics)
        return metrics

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=self.batch_size, shuffle=False, num_workers=4)
