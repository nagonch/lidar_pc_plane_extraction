import os
from torch.utils.data import Dataset
import torch
import numpy as np
import numba as nb


class KittiDataset(Dataset):
    def __init__(self, filenames,
                 scene_size=120000, n_classes=2, keep_road=False, return_instance=False):
        self.scene_size = scene_size
        self.keep_road = keep_road
        self.map_classes = np.vectorize(
            self.get_map,
        )
        self.n_classes = n_classes
        self.filenames = filenames
        self.return_instance = return_instance

    def get_map(self, x):
        if self.keep_road:
            class_map = {40: 2}
        else:
            class_map = {40: 1}
        return class_map.get(x, 0)

    def read_labels(self, filename, filename_manual, filename_instance=None):
        labels_road = (np.fromfile(filename, dtype=np.int32) & 0xFFFF).reshape((-1, 1)).astype(np.uint8)
        labels_road = self.map_classes(labels_road)[:, 0]
        labels_plane = np.load(filename_manual)
        instance_labels = None
        if filename_instance:
            instance_labels = np.load(filename_instance)
        result_labels = np.clip(labels_road + labels_plane, 0, int(self.keep_road) + 1).astype(np.uint8)

        return torch.from_numpy(result_labels[None].T), instance_labels
    
    def read_scene(self, filename):
        np_array = np.fromfile(
            filename, 
            dtype=np.float32
        ).reshape(-1, 4)

        return torch.from_numpy(np_array)

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filenames = self.filenames[idx]
        scene_path = filenames[0]
        labels_path = filenames[1]
        manual_path = filenames[2]
        if self.return_instance:
            instance_path = filenames[3]
        else:
            instance_path = None
        scene = self.read_scene(scene_path)
        labels, instance_labels = self.read_labels(labels_path, manual_path, instance_path)
        indices = np.random.choice(range(len(labels)),
                                size=self.scene_size)
        result = (scene[indices], labels[indices])
        if self.return_instance:
            result += (instance_labels[indices],)
        return result


def get_kitti_filepaths(
    train_size,
    drives=['00'],
    scenes_path='/mnt/vol0/datasets/kitti_dataset/sequences/{}/velodyne',
    labels_original='/mnt/vol0/datasets/kitti_dataset/sequences/{}/labels',
    labels_manual='/mnt/vol0/datasets/kitti_dataset/sequences/{}/manual_labels',
    labels_instance='/mnt/vol0/datasets/kitti_dataset/sequences/{}/instance_labels',
    return_instance=False,
):
    labels = []
    scenes = []
    manual_labels = []
    instance_labels = []
    for drive in drives:
        folder_manual_labels = [labels_manual.format(drive) + '/' + scene for scene in os.listdir(labels_manual.format(drive))]
        folder_instance_labels = [labels_instance.format(drive) + '/' + scene for scene in os.listdir(labels_manual.format(drive))]
        folder_scenes = [scenes_path.format(drive) + '/' + scene.split("/")[-1].replace('.npy', '.bin') for scene in folder_manual_labels]
        folder_labels = [labels_original.format(drive) + '/' + scene.split("/")[-1].replace('.npy', '.label') for scene in folder_manual_labels]
        labels.append(folder_labels)
        scenes.append(folder_scenes)
        manual_labels.append(folder_manual_labels)
        instance_labels.append(folder_instance_labels)
    
    manual_labels = [label for folder in manual_labels for label in folder]
    labels = [label for folder in labels for label in folder]
    scenes = [scene for folder in scenes for scene in folder]
    instance_labels = [label for folder in instance_labels for label in folder]
    
    result = []
    for scene, label, manual_label, instance_label in zip(scenes, labels, manual_labels, instance_labels):
        lis = [scene, label, manual_label]
        if return_instance:
            lis += instance_label
        result.append(lis)
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

