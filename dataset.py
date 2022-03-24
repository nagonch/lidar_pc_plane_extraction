import os
from torch.utils.data import Dataset
import torch
import numpy as np
import numba as nb
import open3d as o3d

class KittiDataset(Dataset):
    def __init__(self, filenames,
                 scene_size=19000, n_classes=2, keep_road=False):
        self.scene_size = scene_size
        self.keep_road = keep_road
        self.map_classes = np.vectorize(
            self.get_map,
        )
        self.n_classes = n_classes
        self.filenames = filenames

    def get_map(self, x):
        if self.keep_road:
            class_map = {40: 2}
        else:
            class_map = {40: 1}
        return class_map.get(x, 0)

    def read_labels(self, filename):
        labels = np.load(filename)
        unique_labels = np.unique(labels)
        substitute_labels = np.arange(unique_labels.shape[0]) + 1
        mapping = dict(zip(unique_labels, substitute_labels))
        map_vector = np.vectorize(
            mapping.get,
        )
        labels = map_vector(labels)
        return torch.from_numpy(labels)
        
    def read_scene(self, filename):
        np_array = np.load(
            filename, 
        ).reshape(-1, 3)
        
        return torch.from_numpy(np_array)

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        scene_path, labels_path = self.filenames[idx]
        scene = self.read_scene(scene_path)[:self.scene_size]
        labels = self.read_labels(labels_path)[:self.scene_size]
        return scene, labels.reshape(-1, 1)


def get_kitti_filepaths(
    train_size,
    drives=['00'],
    scenes_path='/mnt/vol0/datasets/karla_dataset/',
):
    labels = []
    scenes = []
    for file in os.listdir(scenes_path):
        path = scenes_path + file
        if path.endswith('-scene.npy'):
            scenes.append(path)
            labels.append(path.replace('-scene.npy', '.npy'))
    result = []
    for scene, label in zip(scenes, labels):
        result.append([scene, label])
    train = result[:int(len(result) * train_size)]
    test = result[int(len(result) * train_size):]
    return train, test


def get_carla_filepaths(train_size):
    'Not implemented'
    result = []
    train = result[:int(len(result) * train_size)]
    test = result[int(len(result) * train_size):]
    return train, test


def get_filepaths(
    dataset,
    train_size,
):
    train_paths = []
    val_paths = []
    if dataset in ['kitti', 'both']:
        kitti_train, kitti_val = get_kitti_filepaths(train_size)
        train_paths.extend(kitti_train)
        val_paths.extend(kitti_val)

    if dataset in ['carla', 'both']:
        carla_train, carla_test = get_carla_filepaths(train_size)
        train_paths.extend(carla_train)
        val_paths.extend(carla_test)

    if not dataset in ['carla', 'both', 'kitti']:
        raise NotImplementedError(f'Dataset config option {dataset} not implemented')
    
    return train_paths, val_paths

