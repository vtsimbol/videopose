# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import cv2


def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints * joints_vis, joints_vis


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR
    )

    return dst_img


def yes_or_no(p=0.5):
    return np.random.uniform() < p


def blur(img):
    if not yes_or_no():
        return img

    if yes_or_no():
        blur_type = np.random.randint(0, 3)
        if blur_type == 0:
            kernel = int(np.random.uniform(1, 10))
            img_t = cv2.blur(img.copy(), (kernel, kernel))
        elif blur_type == 1:
            kernels = [5, 15, 25]
            kernel = np.random.randint(0, len(kernels))
            kernel = kernels[kernel]
            img_t = cv2.GaussianBlur(img.copy(), (kernel, kernel), 0)
        else:
            img_t = cv2.bilateralFilter(img.copy(), int(np.random.uniform(5, 15)), 75, 75)
    else:
        min_size = min(img.shape[:2])
        kernel_size = np.random.randint(1 + int(min_size * 0.005), 2 + int(min_size * 0.01))

        kernel = np.zeros((kernel_size, kernel_size))
        if np.random.randint(0, 2):
            kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
        else:
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

        kernel /= kernel_size
        img_t = cv2.filter2D(img.copy(), -1, kernel)

    return img_t


def flip(img, joints, joints_vis, center, scale, flip_pairs):
    if not yes_or_no():
        return img, joints, joints_vis, center, scale

    h, w = img.shape[:2]
    if yes_or_no():
        rotate_mode = np.random.randint(0, 3)
        if rotate_mode == 0:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            joints = np.asarray([(h - j[1], j[0], j[2]) for j in joints])
            center[0], center[1] = h - center[1], center[0]
            scale[0], scale[1] = scale[1], scale[0]
        elif rotate_mode == 1:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            joints = np.asarray([(j[1], w - j[0], j[2]) for j in joints])
            center[0], center[1] = center[1], w - center[0]
            scale[0], scale[1] = scale[1], scale[0]
        elif rotate_mode == 2:
            img = cv2.rotate(img, cv2.ROTATE_180)
            joints = np.asarray([(w - j[0], h - j[1], j[2]) for j in joints])
            center[0], center[1] = w - center[0], h - center[1]
    else:
        flip_mode = np.random.randint(0, 2)
        if flip_mode == 0:
            img = cv2.flip(img, flip_mode)
            joints = np.asarray([(j[0], h - j[1], j[2]) for j in joints])
            center[0], center[1] = center[0], h - center[1]
        elif flip_mode == 1:
            img = cv2.flip(img, flip_mode)
            joints = np.asarray([(w - j[0], j[1], j[2]) for j in joints])
            center[0], center[1] = w - center[0], center[1]

        for pair in flip_pairs:
            joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()
            joints_vis[pair[0], :], joints_vis[pair[1], :] = joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return img, joints, joints_vis, center, scale


def crop_frame(img, joints, joints_vis, center, scale, pixel_std=200):
    if not yes_or_no():
        return img, joints, joints_vis, center, scale

    h, w = img.shape[:2]
    roi = np.asarray([0, 0, w, h], dtype=np.float32)

    scale_t = scale * pixel_std
    bbox = np.asarray([np.max([0, center[0] - scale_t[0] * 0.5]),
                       np.max([0, center[1] - scale_t[1] * 0.5]),
                       np.min([center[0] + scale_t[0] * 0.5, w]),
                       np.min([center[1] + scale_t[1] * 0.5, h])], dtype=np.float32)

    crop_side = np.random.randint(0, 4)
    if crop_side == 0:
        roi[0] = np.ceil(np.random.uniform() * bbox[0])
    elif crop_side == 1:
        roi[1] = np.ceil(np.random.uniform() * bbox[1])
    elif crop_side == 2:
        roi[2] -= np.floor(np.random.uniform() * (w - bbox[2]))
    else:
        roi[3] -= np.floor(np.random.uniform() * (h - bbox[3]))

    roi = np.asarray(roi, dtype=int)
    img_t = img[roi[1]:roi[3], roi[0]:roi[2]]
    if img_t.shape[0] == 0 or img_t.shape[1] == 0:
        return img, joints, joints_vis, center, scale

    bbox_t = np.asarray([np.max([bbox[0], roi[0]]),
                         np.max([bbox[1], roi[1]]),
                         np.min([bbox[2], roi[2]]),
                         np.min([bbox[3], roi[3]])], dtype=np.float32)

    joints[:, 0] -= roi[0]
    joints[:, 1] -= roi[1]
    center_t = np.asarray([(bbox_t[0] + bbox_t[2]) / 2, (bbox_t[1] + bbox_t[3]) / 2], dtype=center.dtype)
    scale_t = np.asarray([bbox_t[2] - bbox_t[0], bbox_t[3] - bbox_t[1]], dtype=scale.dtype) / pixel_std

    return img_t, joints, joints_vis, center_t, scale_t


if __name__ == '__main__':
    img_path = '/home/igor/datasets/byndyu_ankle_mobility_heel_coco_format/images/val/000000000005.jpg'
    center = np.asarray([417.0, 421.0])
    scale = np.asarray([593.0 / 200, 823.0 / 200])
    joints = np.asarray([[334, 784, 1], [537, 781, 1]])
    joints_vis = np.asarray([[1, 1, 0], [1, 1, 0]])

    img = cv2.imread(img_path)
    while True:
        viz, joints_t, joints_viz_t, center_t, scale_t = crop_frame(img.copy(), joints.copy(), joints_vis.copy(),
                                                                    center.copy(), scale.copy())
        bbox = np.asarray([center_t[0] - scale_t[0] * 100,
                           center_t[1] - scale_t[1] * 100,
                           center_t[0] + scale_t[0] * 100,
                           center_t[1] + scale_t[1] * 100], dtype=int)

        viz = cv2.rectangle(viz, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        viz = cv2.circle(viz, (int(joints_t[0][0]), int(joints_t[0][1])), 2, (0, 255, 0), 2)
        viz = cv2.circle(viz, (int(joints_t[1][0]), int(joints_t[1][1])), 2, (0, 0, 255), 2)
        cv2.imshow('debug', viz)
        cv2.waitKey(0)
