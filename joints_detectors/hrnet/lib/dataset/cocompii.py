from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.utils.data as data

from dataset.JointsDataset import JointsDataset
from dataset.coco import COCODataset
from dataset.mpii import MPIIDataset


class CocoMpii(data.Dataset):
    num_joints = 13
    coco_ids = [
        (0, 0, 'head'),

        (1, 5, 'l_shoulder'),
        (2, 6, 'r_shoulder'),

        (3, 7, 'l_elbow'),
        (4, 8, 'r_elbow'),

        (5, 9, 'l_wirst'),
        (6, 10, 'r_wirst'),

        (7, 11, 'l_hip'),
        (8, 12, 'r_hip'),

        (9, 13, 'l_knee'),
        (10, 14, 'r_knee'),

        (11, 15, 'l_ankle'),
        (12, 16, 'r_ankle'),
    ]
    mpii_ids = [
        (0, 9, 'head'),

        (1, 13, 'l_shoulder'),
        (2, 12, 'r_shoulder'),

        (3, 14, 'l_elbow'),
        (4, 11, 'r_elbow'),

        (5, 15, 'l_wirst'),
        (6, 10, 'r_wirst'),

        (7, 3, 'l_hip'),
        (8, 2, 'r_hip'),

        (9, 4, 'l_knee'),
        (10, 1, 'r_knee'),

        (11, 5, 'l_ankle'),
        (12, 0, 'r_ankle'),
    ]
    flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]

    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self._datasets = OrderedDict()
        for i, r in enumerate(root):
            if 'coco' in r:
                self._datasets['coco'] = COCODataset(cfg, r, image_set[i], is_train, transform)
            elif 'mpii' in r:
                self._datasets['mpii'] = MPIIDataset(cfg, r, image_set[i], is_train, transform)
            else:
                raise NotImplementedError('Unknown dataset')

        self._lens = OrderedDict({k: len(self._datasets[k]) for k in self._datasets.keys()})

    def __getitem__(self, index):
        key, local_i = self._parse_index(index)
        input, target, target_weight, meta = self._datasets[key][local_i]

        new_meta = meta.copy()
        new_meta['joints'] = np.zeros((self.num_joints, 3), dtype=np.float32)
        new_meta['joints_vis'] = np.zeros((self.num_joints, 3), dtype=np.float32)
        new_meta['center'] = np.array(meta['center'], dtype=np.float32)
        new_meta['scale'] = np.array(meta['scale'], dtype=np.float32)

        _, h, w = target.shape
        new_target = torch.zeros((self.num_joints, h, w), dtype=target.dtype)
        new_target_weight = torch.zeros((self.num_joints, 1), dtype=target_weight.dtype)
        if key == 'coco':
            for i_new, i_source, _ in self.coco_ids:
                new_target[i_new] = target[i_source]
                new_target_weight[i_new] = target_weight[i_source]
                new_meta['joints'][i_new] = meta['joints'][i_source]
                new_meta['joints_vis'][i_new] = meta['joints_vis'][i_source]
        elif key == 'mpii':
            for i_new, i_source, _ in self.mpii_ids:
                new_target[i_new] = target[i_source]
                new_target_weight[i_new] = target_weight[i_source]
                new_meta['joints'][i_new] = meta['joints'][i_source]
                new_meta['joints_vis'][i_new] = meta['joints_vis'][i_source]

        return input, new_target, new_target_weight, new_meta

    def __len__(self):
        return sum([self._lens[k] for k in self._lens.keys()])

    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path, *args, **kwargs):
        offset = 0
        name_values = {}
        perf_indicator = {}
        for k in self._datasets.keys():
            if k == 'mpii':
                print('MPII evaluate')
                new_preds = np.zeros((self._lens[k], self._datasets[k].num_joints, 3), dtype=preds.dtype)
                for i_person in range(self._lens[k]):
                    for i_new, i_source, _ in self.mpii_ids:
                        new_preds[i_person, i_source, :] = preds[i_person + offset, i_new, :]
                name_values[k], perf_indicator[k] = self._datasets[k].evaluate(cfg, new_preds, output_dir)
                offset += self._lens[k]
            elif k == 'coco':
                print('COCO evaluate')
                new_preds = np.zeros((self._lens[k], self._datasets[k].num_joints, 3), dtype=preds.dtype)
                for i_person in range(self._lens[k]):
                    for i_new, i_source, _ in self.coco_ids:
                        new_preds[i_person, i_source, :] = preds[i_person + offset, i_new, :]
                name_values[k], perf_indicator[k] = self._datasets[k].evaluate(cfg, new_preds, output_dir,
                                                                               all_boxes[offset:offset + self._lens[k], :],
                                                                               img_path[offset:offset + self._lens[k]])
                offset += self._lens[k]

        return name_values, perf_indicator

    def _parse_index(self, index):
        keys = list(self._lens.keys())
        key = keys[0]
        for i, k in enumerate(keys):
            l = self._lens[k]
            if index >= l:
                key = keys[i + 1]
                index -= l
            else:
                key = k
                break

        return key, index
