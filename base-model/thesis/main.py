import os
from utils.config import *
from random import randint, seed
import open3d as o3d

from datasets.scannet import ScanNet
from datasets.data_modules import ScanNetSegmentationDataModule
from training.train import train

#seed(1234)
ROOT = get_project_root()


def cli_main():
    config = get_config(f'{ROOT}/cfgs/config.yaml')
    train(config)
    # visualize_random_point_cloud(
    #    ScanNet(f'{ROOT}/{config.SCANNET_CONFIG_PATH}', 'train'))


def visualize_random_point_cloud(dataset):
    idx = randint(0, len(dataset))
    print(f'Visualizing scene: {idx}')
    o3d.visualization.draw_geometries(
        [dataset[idx]])


if __name__ == "__main__":
    cli_main()
