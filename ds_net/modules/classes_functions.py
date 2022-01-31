import numpy as np
import random

CLASS_MAP = {
    0: 0,
    1: 0,
    10: 1,
    11: 0,
    13: 1,
    15: 0,
    16: 1,
    18: 1,
    20: 1,
    30: 0,
    31: 0,
    32: 0,
    40: 1,
    44: 1,
    48: 1,
    49: 1,
    50: 1,
    51: 1,
    52: 1,
    60: 0,
    70: 0,
    71: 0,
    72: 0,
    80: 0,
    81: 0,
    99: 0,
    252: 1,
    253: 0,
    254: 0,
    255: 0,
    256: 1,
    257: 1,
    258: 1,
    259: 1,
}
LABELS_PATH = "/mnt/vol0/datasets/kitti_dataset/sequences/{}/labels/{}.label"
SCENES_PATH = "/mnt/vol0/datasets/kitti_dataset/sequences/{}/velodyne/{}.bin"
DRIVE_NAMES = sorted([f"{i}{j}" for j in range(10) for i in range(2)],
                      key=lambda x: int(x))[:11]
DRIVE_SCENES = [
    4541, 1101, 4661, 801,
    271, 2250, 1101, 1101,
    4071, 1591, 1201, 921,
    1061, 3281, 631, 1901,
    1731, 491, 1801, 4981,
    831, 2721,
][:5]
DRIVE_TO_SCENE_NUMBER = dict(zip(DRIVE_NAMES, DRIVE_SCENES))


def get_zero_padding(num, max_len):
    num_zeros = max_len - len(str(num))
    if num_zeros >= 0:
        return "0" * num_zeros + str(num)
    else:
        return


def get_dataset_elements_list(random_shuffle=False, random_seed=42):
    elements = []
    for key, value in DRIVE_TO_SCENE_NUMBER.items():
        for i in range(value):
            filenum = get_zero_padding(i, 6)
            pair = [SCENES_PATH.format(key, filenum),
                    LABELS_PATH.format(key, filenum)]
            elements.append(pair)
    if random_shuffle:
        random.Random(random_seed).shuffle(elements)
    return elements
