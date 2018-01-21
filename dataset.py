import os
import random
import numpy as np
import pandas as pd
import torch.utils.data as data
from PIL import Image  # Replace by accimage when ready
from random import randint as rint


def make_ego_dataset(root):
    data_path = os.path.join(root, 'images')
    label_path = os.path.join(root, 'labels')
    clips = []
    # make clips
    subject_dirs = os.listdir(data_path)
    for subject_dir in subject_dirs:
        subjcet_path = os.path.join(data_path, subject_dir)
        label_subject_path = os.path.join(label_path, subject_dir)
        scene_dirs = os.listdir(subjcet_path)
        for scene_dir in scene_dirs:
            scene_path = os.path.join(subjcet_path, scene_dir)
            label_scene_path = os.path.join(label_subject_path, scene_dir)
            rgb_dirs = sorted(os.listdir(os.path.join(scene_path, 'Color')))
            for rgb_dir in rgb_dirs:
                rgb_path = os.path.join(scene_path, 'Color', rgb_dir)
                label_csv = os.path.join(label_scene_path, 'Group' + rgb_dir[-1] + '.csv')
                # print("now for data dir %s" % rgb_path)
                f = pd.read_csv(label_csv, header=None)
                img_dirs = sorted(os.listdir(rgb_path))
                for i in range(len(f)):
                    clip = []
                    label, begin, end = f.iloc[i]
                    # add more frame
                    if begin - 2 >= 0:
                        clip.append(os.path.join(rgb_path, img_dirs[begin - 2]))
                        clip.append(os.path.join(rgb_path, img_dirs[begin - 1]))
                    for j in range(end - begin):
                        clip.append(os.path.join(rgb_path, img_dirs[begin + j]))
                    if end + 1 < len(img_dirs):
                        clip.append(os.path.join(rgb_path, img_dirs[end]))
                        clip.append(os.path.join(rgb_path, img_dirs[end + 1]))
                    clips.append((clip, int(label) - 1))
    return clips


class EGOImageFolder(data.Dataset):
    def __init__(self, mode, root):
        self.resize_shape = [48, 120]  # lh
        self.crop_shape = [32, 112, 112]
        self.mode = mode
        self.mean = (114.31, 123.11, 125.56)
        self.std = (38.7568578, 37.88248729, 40.02898126)
        if mode == 'train':
            root = os.path.join(root, 'train')
        elif mode == 'val':
            root = os.path.join(root, 'val')
        elif mode == 'eval':
            root = os.path.join(root, 'val')
            self.resize_shape = [[48, 120], [64, 180]]
        self.clips = make_ego_dataset(root)
        print('clips prepare finished for ', root)

    def __getitem__(self, index):
        paths_total, label = self.clips[index]
        if self.mode == 'train':
            clips = train_twby(paths_total, self.resize_shape, self.crop_shape, self.mean, self.std)
            return clips, label
        elif self.mode == 'val':
            clips = val_twby(paths_total, self.resize_shape, self.crop_shape, self.mean, self.std)
            return clips, label
        elif self.mode == 'eval':
            clips = val_twby_multi(paths_total, self.resize_shape, self.crop_shape, self.mean, self.std)
            return clips, label, paths_total[0]
        else:
            raise (RuntimeError("mode not right"))

    def __len__(self):
        return len(self.clips)


def make_twby_dataset(data_root, f):
    clips = []
    # make clips
    for i in range(len(f)):
        img_dirs = os.path.join(data_root, str(f.loc[i, 0]))
        label = f.loc[i, 1]
        imgs = sorted(os.listdir(img_dirs))
        clip = []
        for img in imgs:
            clip.append(os.path.join(img_dirs, img))
        clips.append((clip, int(label)))
    return clips


def train_twby(paths, min_resize, crop_shape, mean, std, use_flip=False):
    img = Image.open(paths[0])
    while len(paths) < min_resize[0]:
        tmp = []
        [tmp.extend([x, x]) for x in paths]
        paths = tmp
    size = [len(paths), img.height, img.width]
    resize_shape = [rint(min_resize[i], size[i]) for i in range(2)]
    resize_shape.append(int(size[2] * resize_shape[1] / size[1]))
    start = [rint(0, resize_shape[i] - crop_shape[i]) for i in range(3)]

    # resize length
    # interval = len(paths) / resize_shape[0]
    # uniform_list = [int(i * interval) for i in range(resize_shape[0])]
    # paths = [paths[x] for x in uniform_list]
    # crop length
    paths = paths[start[0]:start[0] + crop_shape[0]]

    flip_rand = rint(0, 1)
    clip = []
    for path in paths:
        img = Image.open(path)
        # resize和crop是先w再h
        img = img.resize((resize_shape[2], resize_shape[1]))
        box = (start[2], start[1], start[2] + crop_shape[2], start[1] + crop_shape[1])
        img = img.crop(box)
        if use_flip and flip_rand:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = np.array(img, dtype=np.float32)
        img -= mean
        img /= std
        clip.append(img)
    clip = np.array(clip)
    clip = np.transpose(clip, (3, 0, 1, 2))

    return clip


def val_twby(paths, resize_shape, crop_shape, mean, std):
    img = Image.open(paths[0])
    while len(paths) < resize_shape[0]:
        tmp = []
        [tmp.extend([x, x]) for x in paths]
        paths = tmp
    size = [len(paths), img.height, img.width]
    resize_shape.append(int(size[2] * resize_shape[1] / size[1]))
    start = [(resize_shape[i] - crop_shape[i]) // 2 for i in range(3)]

    # resize length
    interval = size[0] / resize_shape[0]
    uniform_list = [int(i * interval) for i in range(resize_shape[0])]
    paths = [paths[x] for x in uniform_list]
    # crop length
    paths = paths[start[0]:start[0] + crop_shape[0]]

    clip = []
    for path in paths:
        img = Image.open(path)
        # resize和crop是先w再h
        img = img.resize((resize_shape[2], resize_shape[1]))
        box = (start[2], start[1], start[2] + crop_shape[2], start[1] + crop_shape[1])
        img = img.crop(box)
        img = np.array(img, dtype=np.float32)
        img -= mean
        img /= std
        clip.append(img)
    clip = np.array(clip)
    clip = np.transpose(clip, (3, 0, 1, 2))

    return clip


def val_twby_multi(paths_total, resize_shape, crop_shape, mean, std, use_flip=False):
    img = Image.open(paths_total[0])
    while len(paths_total) < resize_shape[-1][0]:
        tmp = []
        [tmp.extend([x, x]) for x in paths_total]
        paths_total = tmp
    size = [len(paths_total), img.height, img.width]
    clips = []
    # 不同尺度
    for i in range(len(resize_shape)):
        rshape = resize_shape[i]
        rshape.append(int(size[2] * rshape[1] / size[1]))
        cshape = crop_shape
        gap = [rshape[j] - cshape[j] for j in range(3)]
        starts = [[0, 0, 0], [0, gap[1], 0], [0, 0, gap[2]], [0, gap[1], gap[2]],
                  [gap[0], 0, 0], [gap[0], gap[1], 0], [gap[0], 0, gap[2]], [gap[0], gap[1], gap[2]]]
        # 不同位置
        for start in starts:
            interval = len(paths_total) / rshape[0]
            uniform_list = (int(i * interval) for i in range(rshape[0]))
            paths = [paths_total[x] for x in uniform_list]
            paths = paths[start[0]:start[0] + cshape[0]]
            clip = []
            for path in paths:
                img = Image.open(path)
                img = img.resize((rshape[2], rshape[1]))
                box = (start[2], start[1], start[2] + cshape[2], start[1] + cshape[1])
                img = img.crop(box)
                img = np.array(img, dtype=np.float32)
                img -= mean
                img /= std
                clip.append(img)
            clip = np.array(clip)
            clip = np.transpose(clip, (3, 0, 1, 2))
            clips.append(clip)
            if use_flip:
                clip = np.flip(clip, 3).copy()
                clips.append(clip)

    return clips


class JesterImageFolder(data.Dataset):
    def __init__(self, mode, data_root, csv_root):
        self.mode = mode
        self.classes = [x for x in range(27)]
        self.mean = (115, 122, 125)
        self.std = (58.65, 61.2, 58.65)
        self.resize_shape = [48, 98]
        self.crop_shape = [32, 64, 64]

        if mode == 'train':
            f = pd.read_csv(os.path.join(csv_root, 'train.csv'), header=None)
        elif mode == 'val':
            f = pd.read_csv(os.path.join(csv_root, 'val.csv'), header=None)
        elif mode == 'eval':
            f = pd.read_csv(os.path.join(csv_root, 'val.csv'), header=None)
            self.resize_shape = [[48, 72], [64, 98]]
        else:
            raise (RuntimeError("Data mode error"))
        self.clips = make_twby_dataset(data_root, f)

        if len(self.clips) == 0:
            raise (RuntimeError("Found 0 clips"))

    def __getitem__(self, index):
        paths_total, label = self.clips[index]

        if self.mode == 'train':
            clips = train_twby(paths_total, self.resize_shape, self.crop_shape, self.mean, self.std)
        elif self.mode == 'val':
            clips = val_twby(paths_total, self.resize_shape, self.crop_shape, self.mean, self.std)
        elif self.mode == 'eval':
            clips = val_twby_multi(paths_total, self.resize_shape, self.crop_shape, self.mean, self.std)
            return clips, label, paths_total[0]
        else:
            raise (RuntimeError("Data mode error"))

        return clips, label

    def __len__(self):
        return len(self.clips)


class SomeImageFolder(data.Dataset):
    def __init__(self, mode, data_root, csv_root):
        self.mode = mode
        self.classes = [x for x in range(174)]
        self.mean = (115, 122, 125)
        self.std = (58.65, 61.2, 58.65)
        self.resize_shape = [48, 98]
        self.crop_shape = [32, 64, 64]

        if mode == 'train':
            f = pd.read_csv(os.path.join(csv_root, 'train.csv'), header=None)
        elif mode == 'val':
            f = pd.read_csv(os.path.join(csv_root, 'val.csv'), header=None)
        elif mode == 'eval':
            f = pd.read_csv(os.path.join(csv_root, 'val.csv'), header=None)
            self.resize_shape = [[48, 72], [64, 98]]
        else:
            raise (RuntimeError("Data mode error"))

        self.clips = make_twby_dataset(data_root, f)

        if len(self.clips) == 0:
            raise (RuntimeError("Found 0 clips"))


def __getitem__(self, index):
    paths_total, label = self.clips[index]

    if self.mode == 'train':
        clips = train_twby(paths_total, self.resize_shape, self.crop_shape, self.mean, self.std)
    elif self.mode == 'val':
        clips = val_twby(paths_total, self.resize_shape, self.crop_shape, self.mean, self.std)
    elif self.mode == 'eval':
        clips = val_twby_multi(paths_total, self.resize_shape, self.crop_shape, self.mean, self.std)
        return clips, label, paths_total[0]
    else:
        raise (RuntimeError("Data mode error"))

    return clips, label


def __len__(self):
    return len(self.clips)
