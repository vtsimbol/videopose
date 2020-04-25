from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.utils.data as data

from dataset.JointsDataset import JointsDataset
from dataset.coco import COCODataset
from dataset.mpii import MPIIDataset
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform


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
        new_meta['joints_gt'] = np.zeros((17, 3), dtype=np.float32)

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

        new_meta['joints_gt'][:self._datasets[key].num_joints] = meta['joints_gt'][:self._datasets[key].num_joints]
        return input, new_target, new_target_weight, new_meta

    def __len__(self):
        return sum([self._lens[k] for k in self._lens.keys()])

    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path, all_joints_gt, *args, **kwargs):
        offset = 0
        name_values = {}
        perf_indicator = {}
        for k in self._datasets.keys():
            if k == 'mpii':
                print('MPII evaluate')
                new_preds = np.zeros((self._lens[k], self._datasets[k].num_joints, 3), dtype=preds.dtype)
                for i in range(self._lens[k]):
                    new_preds[i, :, :] = all_joints_gt[i + offset, :self._datasets[k].num_joints, :]
                    for i_new, i_source, _ in self.mpii_ids:
                        new_preds[i, i_source, :] = preds[i + offset, i_new, :]

                    # # debug
                    # img = cv2.imread(img_path[i + offset], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                    # for pts_idx, pts in enumerate(all_joints_gt[i + offset]):
                    #     x, y = int(pts[0]), int(pts[1])
                    #     img = cv2.circle(img, (x, y), 2, (0, 255, 0), 2)
                    #     img = cv2.putText(img, f'{pts_idx}', (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                    # for pts_idx, pts in enumerate(new_preds[i]):
                    #     x, y = int(pts[0]), int(pts[1])
                    #     img = cv2.circle(img, (x, y), 2, (0, 0, 255), 2)
                    #     img = cv2.putText(img, f'{pts_idx}', (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0,  255), 1)
                    # cv2.imshow('debug', img)
                    # cv2.waitKey(0)

                name_values[k], perf_indicator[k] = self._datasets[k].evaluate(cfg, new_preds, output_dir)
                offset += self._lens[k]
            elif k == 'coco':
                print('COCO evaluate')
                new_preds = np.zeros((self._lens[k], self._datasets[k].num_joints, 3), dtype=preds.dtype)
                for i in range(self._lens[k]):
                    new_preds[i, :, :] = all_joints_gt[i + offset, :self._datasets[k].num_joints, :]
                    for i_new, i_source, _ in self.coco_ids:
                        new_preds[i, i_source, :] = preds[i + offset, i_new, :]

                    # # debug
                    # img = cv2.imread(img_path[i + offset], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                    # for pts_idx, pts in enumerate(all_joints_gt[i + offset]):
                    #     x, y = int(pts[0]), int(pts[1])
                    #     img = cv2.circle(img, (x, y), 2, (0, 255, 0), 2)
                    #     img = cv2.putText(img, f'{pts_idx}', (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                    # for pts_idx, pts in enumerate(new_preds[i]):
                    #     x, y = int(pts[0]), int(pts[1])
                    #     img = cv2.circle(img, (x, y), 2, (0, 0, 255), 2)
                    #     img = cv2.putText(img, f'{pts_idx}', (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0,  255), 1)
                    # cv2.imshow('debug', img)
                    # cv2.waitKey(0)

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
