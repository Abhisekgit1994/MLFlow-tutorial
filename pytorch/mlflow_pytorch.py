import os
import pytorch_lightning as pl
import torch
from torch.nn import functional as fn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchmetrics.functional import accuracy

import mlflow.pytorch

from mlflow import MlflowClient


class MNISTModel(pl.LightningModule):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.l1 = torch.nn.Linear(28*28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = fn.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)

        acc = (pred == y).sum()/len(y)

        self.log("train_loss", loss, on_epoch=True)
        self.log("acc", acc, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith('mlflow.')}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id)]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


model = MNISTModel()
train_ds = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=32)

trainer = pl.Trainer(max_epochs=20)

mlflow.pytorch.autolog()

with mlflow.start_run() as run:
    trainer.fit(model, train_loader)

print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
