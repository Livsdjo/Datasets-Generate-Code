#!/usr/bin/env python
"""
Copyright 2017, Zixin Luo, HKUST.
IO tools.
"""

from __future__ import print_function

import os
import re
from struct import unpack
import numpy as np


def load_pfm(pfm_path):
    with open(pfm_path, 'rb') as fin:
        color = None
        width = None
        height = None
        scale = None
        data_type = None
        header = str(fin.readline().decode('UTF-8')).rstrip()

        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', fin.readline().decode('UTF-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        scale = float((fin.readline().decode('UTF-8')).rstrip())
        if scale < 0:  # little-endian
            data_type = '<f'
        else:
            data_type = '>f'  # big-endian
        data_string = fin.read()
        data = np.fromstring(data_string, data_type)
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flip(data, 0)
    return data


def read_corr(file_path):
    """Read the match correspondence file.
    Args:
        file_path: file path.
    Returns:
        matches: list of match data, each consists of two image indices and Nx15 match matrix, of
        which each line consists of two 2x3 transformations, geometric distance and two feature
        indices.
    """
    matches = []
    with open(file_path, 'rb') as fin:
        # print(fin)
        while True:
            rin = fin.read(24)
            # print(rin)
            if len(rin) == 0:
                # EOF
                break
            idx0, idx1, num=unpack('Q' * 3, rin)    # L
            bytes_theta = num * 60
            corr = np.fromstring(fin.read(bytes_theta), dtype=np.float32).reshape(-1, 15)
            matches.append([idx0, idx1, corr])
            # print(idx0, idx1, corr)
        # print([idx0, idx1, corr])
    return matches


def read_corr_1(file_path):
    """Read the match correspondence file.
    Args:
        file_path: file path.
    Returns:
        matches: list of match data, each consists of two image indices and Nx15 match matrix, of
        which each line consists of two 2x3 transformations, geometric distance and two feature
        indices.
    """
    matches = []
    count  = 0
    while True:
        if count <= 2000:
            fin = open('./corr1.bin', 'wb+')
            count += 1
        elif count > 2000 and count <= 4000:
            fin = open('./corr2.bin', 'wb+')
            count += 1
        elif count > 4000 and count <= 6000:
            fin = open('./corr3.bin', 'wb+')
            count += 1
        elif count > 6000 and count <= 8000:
            fin = open('./corr4.bin', 'wb+')
            count += 1
        else:
            fin = open('./corr5.bin', 'wb+')
            count += 1

        rin = fin.read(24)
        # print(rin)
        if len(rin) == 0:
            # EOF
            break
        idx0, idx1, num=unpack('Q' * 3, rin)    # L
        bytes_theta = num * 60
        rin1 = fin.read(bytes_theta)
        corr = np.fromstring(rin1, dtype=np.float32).reshape(-1, 15)
        matches.append([idx0, idx1, corr])

    return matches



def read_kpt(file_path):
    """Read the keypoint file.
    Args:
        file path: file path.
    Returns:
        kpt_data: keypoint data of Nx6 numpy array.
    """
    kpt_data = np.fromfile(file_path, dtype=np.float32)
    kpt_data = np.reshape(kpt_data, (-1, 6))
    return kpt_data


def hash_int_pair(ind1, ind2):
    """Hash an int pair.
    Args:
        ind1: int1.
        ind2: int2.
    Returns:
        hash_index: the hash index.
    """
    assert ind1 <= ind2
    return ind1 * 2147483647 + ind2


def read_mask(file_path, size=32):
    """Read the mask file.
    Args:
        file_path: file path.
        size: mask size.
    Returns:
        mask_dict: mask data in dictionary, indexed by hashed pair index.
    """
    mask_dict = {}
    size = size * size * 2
    record_size = 8 + size

    with open(file_path, 'rb') as fin:
        data = fin.read()
    for i in range(0, len(data), record_size):
        decoded = unpack('2i' + '?' * size, data[i: i + record_size])
        mask = np.array(decoded[2:])
        mask_dict[hash_int_pair(decoded[0], decoded[1])] = mask
    return mask_dict


def read_cams(cam_path):
    """
    Args:
        cam_path: Path to cameras.txt.
    Returns:
        cam_dict: A dictionary indexed by image index and composed of (K, t, R, dist, img_size).
        K - 2x3, t - 3x1, R - 3x3, dist - 1x3, img_size - 1x2.
    """
    cam_data = [i.split(' ') for i in read_list(cam_path)]

    cam_dict = {}
    for i in cam_data:
        i = [float(j) for j in i if j is not '']
        K = np.array([(i[1], i[5], i[3]),
                      (0, i[2], i[4]), (0, 0, 1)])
        t = np.array([(i[6], ), (i[7], ), (i[8], )])
        R = np.array([(i[9], i[10], i[11]),
                      (i[12], i[13], i[14]),
                      (i[15], i[16], i[17])])
        dist = np.array([i[18], i[19], i[20]])
        img_size = np.array([i[21], i[22]])
        cam_dict[i[0]]= (K, t, R, dist, img_size)          # key:图像下标 value:参数
    return cam_dict


def read_list(list_path):
    """Read list."""
    if list_path is None or not os.path.exists(list_path):
        print('Not exist', list_path)
        exit(-1)
    content= open(list_path).read().splitlines()
    return content
