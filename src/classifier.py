import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.init as init
import torchmetrics.classification as metrics
from torch.optim import Adam

import wandb
from dataset import surface_classes

# Use the seborn style
plt.style.use("seaborn")
# But with fonts from the document body
plt.rcParams.update(
    {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
)


class AudioClassifier(L.LightningModule):
    def __init__(self, learning_rate=1e-3, classify_by="spin"):
        super(AudioClassifier, self).__init__()
        self.learning_rate = learning_rate
        self.classify_by = classify_by  # 'surface' or 'spin'
        if classify_by == "spin":
            self.out_features = 3
        elif classify_by == "surface":
            self.out_features = 13

        # Define the convolutional layers
        self.conv = nn.Sequential(
            self._conv_block(1, 2, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            self._conv_block(2, 4, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            self._conv_block(4, 8, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            self._conv_block(8, 16, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            self._conv_block(16, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            self._conv_block(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
        )

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=self.out_features)
        self.sm = nn.Softmax(dim=1)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Defining all the metrics for tracking the model's performances
        self.train_acc = metrics.Accuracy(
            task="multiclass", num_classes=self.out_features
        )
        self.val_acc = metrics.Accuracy(
            task="multiclass", num_classes=self.out_features
        )
        self.train_auroc = metrics.MulticlassAUROC(self.out_features)
        self.val_auroc = metrics.MulticlassAUROC(self.out_features)

        self.train_f1 = metrics.MulticlassF1Score(self.out_features)
        self.val_f1 = metrics.MulticlassF1Score(self.out_features)
        self.train_recall = metrics.MulticlassRecall(self.out_features)
        self.val_recall = metrics.MulticlassRecall(self.out_features)
        self.train_prec = metrics.MulticlassPrecision(self.out_features)
        self.val_prec = metrics.MulticlassPrecision(self.out_features)

        self.val_conf_mat = metrics.MulticlassConfusionMatrix(
            self.out_features, normalize="true"
        )

        self.save_hyperparameters()

    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        """Helper function to create a convolutional block."""
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        relu = nn.ReLU()
        bn = nn.BatchNorm2d(out_channels)

        # Kaiming Initialization
        init.kaiming_normal_(conv.weight, a=0.1)
        conv.bias.data.zero_()

        return nn.Sequential(conv, relu, bn)

    def forward(self, x):
        x = self.conv(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.lin(x)
        x = self.sm(x)
        return x

    def _select_label(self, batch):
        """Select the correct label (surface or spin) based on 'classify_by'."""
        features, surface_class, spin_class = batch
        if self.classify_by == "surface":
            y = surface_class
        else:
            y = spin_class
        return features, y

    def training_step(self, batch, batch_idx):
        x, y = self._select_label(batch)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=False)
        self.train_auroc(y_hat, y)
        self.log("train_auroc", self.train_auroc, on_step=True, on_epoch=False)
        self.train_f1(y_hat, y)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=False)
        self.train_recall(y_hat, y)
        self.log("train_recall", self.train_recall, on_step=True, on_epoch=False)
        self.train_prec(y_hat, y)
        self.log("train_precision", self.train_prec, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        # print(batch)
        x, y = self._select_label(batch)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        self.val_acc(y_hat, y)
        self.log("val_acc", self.val_acc, on_step=True, on_epoch=True)
        self.val_auroc(y_hat, y)
        self.log("val_auroc", self.val_auroc, on_step=True, on_epoch=True)
        self.val_f1(y_hat, y)
        self.log("val_f1", self.val_f1, on_step=True, on_epoch=True)
        self.val_recall(y_hat, y)
        self.log("val_recall", self.val_recall, on_step=True, on_epoch=True)
        self.val_prec(y_hat, y)
        self.log("val_precision", self.val_prec, on_step=True, on_epoch=True)

        self.val_conf_mat.update(y_hat, y)
        return loss

    def on_validation_epoch_end(self):
        # Compute and log the confusion matrix at the end of the validation epoch
        confmat = self.val_conf_mat.compute()

        if self.classify_by == "spin":
            labels = ["Backspin", "Nospin", "Topspin"]
        elif self.classify_by == "surface":
            labels = surface_classes

        # Plot confusion matrix using seaborn
        fig = plt.figure(figsize=(10, 7))
        ax = sns.heatmap(
            confmat.cpu().numpy(),
            annot=True,
            fmt=".2g",
            cmap="viridis",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
        )
        ax.tick_params(axis="x", labelrotation=45)
        ax.tick_params(axis="y", labelrotation=45)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        # plt.title("Confusion Matrix")
        plt.show()

        # Log the confusion matrix plot to WandB
        self.logger.experiment.log({"Confusion Matrix": wandb.Image(fig)})

        # Close the plot to avoid memory issues
        plt.close()

        # Reset confusion matrix for the next epoch
        self.val_conf_mat.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
