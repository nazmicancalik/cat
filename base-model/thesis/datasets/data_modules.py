import torch
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
from pytorch3d.datasets import ShapeNetCore

from datasets.scannet import ScanNet
from datasets.shapenet_core import ShapeNetCore
from datasets.modelnet import ModelNetDataset
from datasets.modelnet_dgcnn import ModelNet40

from utils.config import *


class ScanNetSegmentationDataModule(LightningDataModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.batch_size = config.training.batch_size
        self.num_classes = config.datasets.scannet.num_classes
        self.num_points = config.datasets.scannet.num_points
        self.num_workers = config.training.num_workers

    def train_dataloader(self):
        dataset = ScanNet(self.config, 'train')
        # TODO: Look into the shuffle and the drop last later.
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        dataset = ScanNet(self.config, 'val')
        # TODO: Look into the shuffle and the drop last later.
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)

    def test_dataloader(self):
        dataset = ScanNet(self.config, 'test')
        # TODO: Look into the shuffle and the drop last later.
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)


class ShapeNetDataModule(LightningDataModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.batch_size = config.training.batch_size
        self.num_workers = config.training.num_workers

    def train_dataloader(self):
        dataset = ShapeNetCore(self.config, 'train')
        # TODO: Look into the shuffle and the drop last later.
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        dataset = ShapeNetCore(self.config, 'val')
        # TODO: Look into the shuffle and the drop last later.
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)

    def test_dataloader(self):
        dataset = ShapeNetCore(self.config, 'test')
        # TODO: Look into the shuffle and the drop last later.
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)


class ModelNetDataModule(LightningDataModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.batch_size = config.training.batch_size
        self.num_workers = config.training.num_workers

    def train_dataloader(self):
        #dataset = ModelNetDataset(self.config, 'train')
        dataset = ModelNet40(self.config.datasets.modelnet.num_points, 'train')
        # TODO: Look into the shuffle and the drop last later.
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        #dataset = ModelNetDataset(self.config, 'test')
        dataset = ModelNet40(self.config.datasets.modelnet.num_points, 'test')
        # TODO: Look into the shuffle and the drop last later.
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)

    def test_dataloader(self):
        #dataset = ModelNetDataset(self.config, 'test')
        dataset = ModelNet40(self.config.datasets.modelnet.num_points, 'test')
        # TODO: Look into the shuffle and the drop last later.
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)
