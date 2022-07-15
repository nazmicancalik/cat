'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
Code taken from Xu Yan
'''
import os
import numpy as np
import warnings
import pickle
import math
import random

from tqdm import tqdm
from torch.utils.data import Dataset

from utils.config import *

warnings.filterwarnings('ignore')
#np.random.seed(42)

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


ROOT = get_project_root()


class ModelNetDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.root = f"{ROOT}/{config.datasets.modelnet.root}"
        self.npoints = config.datasets.modelnet.num_points
        self.process_data = config.datasets.modelnet.process_data
        self.uniform = config.datasets.modelnet.use_uniform_sample
        self.use_normals = config.datasets.modelnet.use_normals
        self.num_category = config.datasets.modelnet.num_classes
        self.val_percentage = config.datasets.modelnet.val_percentage
        self.splits_folder = f"{ROOT}/{config.datasets.modelnet.splits_folder}"

        # Create the validation split if it doesn't exist

        if self.num_category == 10:
            if not os.path.exists(os.path.join(self.splits_folder, 'modelnet10_val.txt')) and config.datasets.modelnet.create_val_split:
                print("Creating the validation split as it doesn't exist")
                self._create_val_split()
            self.catfile = os.path.join(
                self.splits_folder, 'modelnet10_shape_names.txt')
        else:
            if not os.path.exists(os.path.join(self.splits_folder, 'modelnet40_val.txt')) and config.datasets.modelnet.create_val_split:
                print("Creating the validation split as it doesn't exist")
                self._create_val_split()
            self.catfile = os.path.join(
                self.splits_folder, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(
                os.path.join(self.splits_folder, 'modelnet10_train.txt'))]
            #shape_ids['val'] = [line.rstrip() for line in open(os.path.join(self.splits_folder, 'modelnet10_val.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(
                os.path.join(self.splits_folder, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(
                os.path.join(self.splits_folder, 'modelnet40_train.txt'))]
            #shape_ids['val'] = [line.rstrip() for line in open(os.path.join(self.splits_folder, 'modelnet40_val.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(
                os.path.join(self.splits_folder, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test' or split == 'val')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))


        if self.uniform:
            self.save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts_fps.dat' % (
                self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts.dat' % (
                self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' %
                      self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(
                        fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(
                            point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def _create_val_split(self):
        shape_ids = [line.rstrip() for line in open(os.path.join(self.splits_folder, f"modelnet{self.num_category}_train_unprocessed.txt"))]
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids]
        shapes = {}
        for id in shape_ids:
            shapename = '_'.join(id.split('_')[0:-1])
            shapes.setdefault(shapename, []).append(id)
        
        # split the items
        train_all_ids = []
        val_all_ids = []
        for classname, ids in shapes.items():
            train_len = int(np.floor(np.abs(1-self.val_percentage) * len(ids)))
            permuted_ids = np.random.permutation(ids)
            
            train_ids = permuted_ids[:train_len]
            val_ids = permuted_ids[train_len:]
            
            train_all_ids += list(train_ids)
            val_all_ids += list(val_ids)

        # write the ids to respective files
        with open(f"{self.splits_folder}/modelnet{self.num_category}_train.txt", "w") as outfile:
            outfile.write("\n".join(str(item) for item in train_all_ids))
        with open(f"{self.splits_folder}/modelnet{self.num_category}_val.txt", "w") as outfile:
            outfile.write("\n".join(str(item) for item in val_all_ids))
        
    def __len__(self):
        return len(self.datapath)

    # Rotates the point cloud in z axis with a random degree
    def _rotation(self,point_set):
        theta = -math.pi/2 +  random.random() * math.pi
        rot_matrix = np.array([
            [ math.cos(theta), -math.sin(theta),    0],
            [ math.sin(theta),  math.cos(theta),    0],
            [0,                             0,      1]])
        rot_pointcloud = rot_matrix.dot(point_set.T).T
        return rot_pointcloud


    def _apply_augmentation(self,point_set):
        # Rotation 
        rand_val = random.random()
        if rand_val < self.config.augmentation.rotation_prob:
            point_set = self._rotation(point_set)

        # Gaussian Noise
        rand_val = random.random()
        if rand_val < self.config.augmentation.noise_prob:
            noise = np.random.normal(0, self.config.augmentation.noise_std, (point_set.shape))
            point_set = point_set + noise

        # Scale
        rand_val = random.random()
        if rand_val < self.config.augmentation.scale_prob:
            scale_factor = random.random() * 0.4 + 0.8 # scale limits [ 0.8 - 1.2 ]
            point_set = point_set * scale_factor

        # Perturbation applied always
        point_set = np.random.permutation(point_set)
        return point_set

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        # apply augmentation depending on the probabilities randomly each time
        point_set[:, 0:3] = self._apply_augmentation(point_set[:,0:3])
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)
