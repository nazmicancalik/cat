from pathlib import Path
from plyfile import PlyData, PlyElement
import open3d as o3d
import numpy as np
import json
import torch

from utils.config import *


ROOT = get_project_root()


class ScanNet(torch.utils.data.Dataset):
    def __init__(self, config, split) -> None:

        # Read the config to set up the dataset.
        self.num_points = config.datasets.scannet.num_points

        assert split in ['train', 'val', 'test', 'overfit']
        if split == 'train':
            self.dataset_root = Path(ROOT, config.datasets.scannet.root_train)
        elif split == 'val':
            self.dataset_root = Path(ROOT, config.datasets.scannet.root_val)
        elif split == 'test':
            self.dataset_root = Path(ROOT, config.datasets.scannet.root_test)

        # Scenes to use / dataset items - ids
        self.items = Path(
            f"{ROOT}/{config.datasets.scannet.splits_folder}/scannetv2_{split}.txt").read_text().splitlines()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item_id = self.items[idx]
        ply_file_id = f'{self.dataset_root}/{item_id}/{item_id}'
        point_cloud = self.get_point_cloud(ply_file_id)
        downsampled_point_cloud = self.downsample_and_normalize_point_cloud(
            point_cloud)
        points = downsampled_point_cloud['points']
        labels = downsampled_point_cloud['labels']
        return self.normalize_point_cloud(points), labels

    def downsample_and_normalize_point_cloud(self, point_cloud: dict) -> dict:
        pc_length = len(point_cloud['points'])
        selected_indices = np.random.randint(
            0, pc_length, self.num_points)
        return {
            'points': point_cloud['points'][selected_indices],
            'labels': point_cloud['labels'][selected_indices]
        }

    def normalize_point_cloud(self, points):
        points /= np.max(np.abs(points), axis=0)
        return points

    @staticmethod
    def get_point_cloud(ply_file_id: str) -> dict:
        point_cloud = o3d.io.read_point_cloud(f'{ply_file_id}_vh_clean_2.ply')
        labels = PlyData.read(f'{ply_file_id}_vh_clean_2.labels.ply')
        labels = labels['vertex']['label']

        # TODO: Maybe use the normals later as input later

        return {
            'points': np.asarray(point_cloud.points),
            'labels': np.asarray(labels).astype(np.int16)
        }
