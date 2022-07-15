import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from modules.dgcnn import LightningDGCNNFeatureExtractor
from thesis.modules.transformer.positional_encoder import LightningPositionalEncoder
from thesis.modules.transformer.transformer import LightningTransformer
from utils.config import *


class LightningSegmentationModule(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.gpus = config.training.gpus
        self.batch_size = config.training.batch_size

        self.max_epochs = config.training.max_epochs
        self.optimizer = config.training.optimizer
        self.learning_rate = config.training.learning_rate

        self.num_classes = config.datasets.scannet.num_classes
        self.num_points = config.datasets.scannet.num_points

        self.seq_len = config.modules.transformer.seq_len
        self.embed_dim = config.modules.transformer.embed_dim

        # Module pieces:
        # Feature extractor to get the features of the downsampled and normalized point cloud.
        # Positional Encoder to encode the positions before feeding it to the transformer.
        # Transformer: Main model to do the point cloud understanding
        # Classifier: Classification head for the task
        self.feature_extractor = LightningDGCNNFeatureExtractor(config)
        self.positional_encoder = LightningPositionalEncoder(config)
        self.transformer = LightningTransformer(config)
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)

        # Might change this later
        self.loss_criterion = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        points, labels = batch

        # This transpose is needed for the inner working principle for the knn calculation.
        points = points.transpose(2, 1).float()
        pos = self.positional_encoder(points)
        print('\n\n pos encoding size:', pos.size(), '\n\n')

        coor, features = self.feature_extractor(points)
        print('\n\n feature size dgcnn:', features.size(), '\n\n')

        transformer_hidden_state = self.transformer(features.transpose(-1, -2))
        print('\n\n transformer output :',
              transformer_hidden_state.size(), '\n\n')

        # Max Pooling of the output hidden state.
        result = transformer_hidden_state.max(dim=2, keepdim=False)[0]

        # Upscale the hidden state with Fold Net or sth similar.
        # TODO

        # Get the mean class based loss. Average loss for each category of object.
        # TODO

        print(result, labels)
        print(result.size(), labels.size())
        loss = self.loss_criterion(result, labels.float())
        loss = 0.5  # for debug purposes

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return {"loss": loss}

    def training_epoch_end(self, training_step_outputs):
        mean_corrects = []
        for out in training_step_outputs:
            # change it to mean correct later
            mean_corrects.append(out["loss"].detach().cpu().clone().numpy())
        train_instance_acc = np.mean(mean_corrects)
        self.log("train_acc", train_instance_acc, on_step=False, on_epoch=True)

    def validaton_step(self, batch, batch_idx):
        # TODO: Implement iou calc here
        pass

    def test_step(self, batch, batch_idx):
        # TODO: Similar to val step
        pass

    def configure_optimizers(self):
        # Add all the parameters of the dgcnn and transformer here
        optimizer = torch.optim.Adam(
            self.feature_extractor.parameters(), lr=self.learning_rate
        )
        return optimizer
