#!/usr/bin/env python3
"""
Copyright 2019, Zixin Luo, HKUST.
Visualization tools.
"""

from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from collections import Counter
import struct

sys.path.append('..')

from utils.geom import get_essential_mat, get_epipolar_dist, undist_points, warp, grid_positions, upscale_positions, \
    downscale_positions, relative_pose
from utils.io import read_kpt, read_corr, read_mask, hash_int_pair, read_cams, load_pfm
from utils.patch_extractor import PatchExtractor
from models import get_model, loc_model
from utils.opencvhelper import MatcherWrapper
import pickle


def draw_kpts(imgs, kpts, color=(0, 255, 0), radius=2, thickness=2):
    """
    Args:
        imgs: color images.
        kpts: Nx2 numpy array.
    Returns:
        all_display: image with drawn keypoints.
    """
    all_display = []
    for idx, val in enumerate(imgs):
        kpt = kpts[idx]
        tmp_img = val.copy()
        for kpt_idx in range(kpt.shape[0]):
            display = cv2.circle(
                tmp_img, (int(kpt[kpt_idx][0]), int(kpt[kpt_idx][1])), radius, color, thickness)
        all_display.append(display)
    all_display = np.concatenate(all_display, axis=1)
    return all_display


def draw_matches(img0, img1, kpts0, kpts1, match_idx,
                 downscale_ratio=1, color=(255, 255, 0), radius=4, thickness=2):
    """
    Args:
        img: color image.
        kpts: Nx2 numpy array.
        match_idx: Mx2 numpy array indicating the matching index.
    Returns:
        display: image with drawn matches.
    """
    resize0 = cv2.resize(
        img0, (int(img0.shape[1] * downscale_ratio), int(img0.shape[0] * downscale_ratio)))
    resize1 = cv2.resize(
        img1, (int(img1.shape[1] * downscale_ratio), int(img1.shape[0] * downscale_ratio)))

    rows0, cols0 = resize0.shape[:2]
    rows1, cols1 = resize1.shape[:2]

    kpts0 *= downscale_ratio
    kpts1 *= downscale_ratio

    display = np.zeros((max(rows0, rows1), cols0 + cols1, 3))
    display[:rows0, :cols0, :] = resize0
    display[:rows1, cols0:(cols0 + cols1), :] = resize1

    if 1:
        for idx in range(match_idx.shape[0]):
            val = match_idx[idx]
            pt0 = (int(kpts0[val[0]][0]), int(kpts0[val[0]][1]))
            pt1 = (int(kpts1[val[1]][0]) + cols0, int(kpts1[val[1]][1]))

            cv2.circle(display, pt0, radius, color, thickness)
            cv2.circle(display, pt1, radius, color, thickness)
            cv2.line(display, pt0, pt1, color, thickness)

    display /= 255

    return display


def draw_mask(img0, img1, mask, size=32, downscale_ratio=1):
    """
    Args:
        img: color image.
        mask: 14x28 mask data.
        size: mask size.
    Returns:
        display: image with mask.
    """
    resize_imgs = []
    resize_imgs.append(cv2.resize(
        img0, (int(img0.shape[1] * downscale_ratio), int(img0.shape[0] * downscale_ratio))))
    resize_imgs.append(cv2.resize(
        img1, (int(img1.shape[1] * downscale_ratio), int(img1.shape[0] * downscale_ratio))))

    imgs_error_area_index = []
    imgs_error_area_index.append(
        np.ones((int(img0.shape[1] * downscale_ratio), int(img0.shape[0] * downscale_ratio))) * 0)
    imgs_error_area_index.append(
        np.ones((int(img1.shape[1] * downscale_ratio), int(img1.shape[0] * downscale_ratio))) * 0)

    masks = []
    masks.append(ndimage.binary_fill_holes(np.reshape(mask[:size * size], (size, size))))
    masks.append(ndimage.binary_fill_holes(np.reshape(mask[size * size:], (size, size))))

    area_count = []
    for idx, val in enumerate(masks):
        # print(idx, val, val.shape, resize_imgs[0].shape)
        h_interval = np.ceil(float(resize_imgs[idx].shape[0]) / val.shape[0])
        w_interval = np.ceil(float(resize_imgs[idx].shape[1]) / val.shape[1])

        count = 0
        for i in range(resize_imgs[idx].shape[0]):
            for j in range(resize_imgs[idx].shape[1]):
                p = int(np.floor(i / h_interval))
                q = int(np.floor(j / w_interval))
                if val[p, q]:
                    count += 1
                    resize_imgs[idx][i, j, 0] = 0
                    imgs_error_area_index[idx][i, j] = 1
        # print("区域面积：", count)
        area_count.append(count)
    display = np.concatenate(resize_imgs, axis=1)
    return display, imgs_error_area_index, area_count


"""
    两个问题：
    1. 正确的匹配用畸变处理的
       错误的匹配没有用畸变处理
    2. 到底行列顺序
"""


class visualize_new_class_def:
    def __init__(self, root, match_pair_idx, blendar=False):
        # read correspondences
        self.root = root
        self.match_pair_idx = match_pair_idx
        corr_path = os.path.join(self.root, 'geolabel', 'corr.bin')
        self.match_records = read_corr(corr_path)
        self.cidx0 = self.match_records[self.match_pair_idx][0]
        self.cidx1 = self.match_records[self.match_pair_idx][1]
        self.basename0 = str(self.cidx0).zfill(8)  # 在左边补齐0 统一长度
        self.basename1 = str(self.cidx1).zfill(8)
        self.match_num = 0
        # print(self.basename0, self.basename1)
        # read images
        if blendar:
            self.img_path0 = os.path.join(self.root, 'blended_images', self.basename0 + '.jpg')
            self.img_path1 = os.path.join(self.root, 'blended_images', self.basename1 + '.jpg')
        else:
            self.img_path0 = os.path.join(self.root, 'undist_images', self.basename0 + '.jpg')
            self.img_path1 = os.path.join(self.root, 'undist_images', self.basename1 + '.jpg')
        self.img0 = cv2.imread(self.img_path0)[..., ::-1]  # 通道颜色顺序可能不一样 https://www.runoob.com/note/51257   变成RGB
        self.img1 = cv2.imread(self.img_path1)[..., ::-1]
        # read cameras
        cam_path = os.path.join(self.root, 'geolabel', 'cameras.txt')
        cam_dict = read_cams(cam_path)
        cam0, cam1 = cam_dict[self.cidx0], cam_dict[self.cidx1]
        self.K0, self.K1 = cam0[0], cam1[0]
        self.t0, self.t1 = cam0[1], cam1[1]
        self.R0, self.R1 = cam0[2], cam1[2]
        self.dist0, self.dist1 = cam0[3], cam1[3]
        self.ori_img_size0, self.ori_img_size1 = cam0[4], cam1[4]

    def save_pkl(self, data_path, data):
        pickle_file = open(data_path, 'wb')  # 创建一个pickle文件，文件后缀名随意,但是打开方式必须是wb（以二进制形式写入）
        pickle.dump(data, pickle_file)  # 将列表倒入文件
        pickle_file.close()

    def get_and_show_right_match(self):
        kpts0 = self.match_records[self.match_pair_idx][2][:, 0:6]
        kpts0 = undist_points(kpts0, self.K0, self.dist0, self.ori_img_size0)

        kpts1 = self.match_records[self.match_pair_idx][2][:, 6:12]
        kpts1 = undist_points(kpts1, self.K0, self.dist1, self.ori_img_size1)

        all_match_kpts_six_dimension_des = np.column_stack((kpts0, kpts1))

        kpts0 = np.stack([kpts0[:, 2], kpts0[:, 5]], axis=-1)
        kpts1 = np.stack([kpts1[:, 2], kpts1[:, 5]], axis=-1)

        sift_match_kpts_normalized_xy_stack = np.column_stack((kpts0, kpts1))
        # validate epipolar geometry  验证对极几何
        e_mat = get_essential_mat(self.t0, self.t1, self.R0, self.R1)
        epi_dist = get_epipolar_dist(kpts0, kpts1, self.K0, self.K1, self.ori_img_size0, self.ori_img_size1, e_mat)
        print('max epipolar distance:', np.max(epi_dist), "正确匹配个数:", len(epi_dist))

        return sift_match_kpts_normalized_xy_stack, all_match_kpts_six_dimension_des

    def epi_detect_error_match(self, all_kpts_list):
        kpts0 = all_kpts_list[0]
        kpts1 = all_kpts_list[1]

        kpts0_complete = undist_points(kpts0, self.K0, self.dist0, self.ori_img_size0)
        kpts0 = np.stack([kpts0_complete[:, 2], kpts0_complete[:, 5]], axis=-1)

        kpts1_complete = undist_points(kpts1, self.K1, self.dist1, self.ori_img_size1)
        kpts1 = np.stack([kpts1_complete[:, 2], kpts1_complete[:, 5]], axis=-1)

        # validate epipolar geometry  验证对极几何
        e_mat = get_essential_mat(self.t0, self.t1, self.R0, self.R1)
        # print(888888)
        # print(kpts0.shape)
        epi_dist = get_epipolar_dist(kpts0, kpts1, self.K0, self.K1, self.ori_img_size0,
                                     self.ori_img_size1, e_mat)
        # print('max epipolar distance', np.max(epi_dist))
        label = np.ones(len(kpts0))

        if 1:
            c = dict(enumerate(epi_dist))
            f = zip(c.values(), c.keys())
            g = sorted(f)
            k = np.array(g)
            print("总的匹配个数:", len(k))
            u = k[k[:, 0] > 0.01][: 2000]  # len(k) - 2000: len(k)        1e-4 0.001

            if len(u) < 2000:
                print("点的个数不够了")
                u = k[k[:, 0] > 0.001][: 2000]  # 0.0001       0.01
            # print("38420849")
            # print(u)
            index = u[:, 1]
            index = index.astype(np.int32)
            label[index] = 0
        else:
            for idx, epi_dist_val in enumerate(epi_dist):
                if (epi_dist_val > 1e-4) and (epi_dist_val < 1e-1):
                    label[idx] = 0

        # -------------------------------------------------
        # 看需不需要极线生成正确匹配了  到时候就看这个阈值设定了  还是很关键这个值 影响很大
        # 因为要模仿其他文章，肯定这里边带有错误的匹配
        # 需要调节一下
        # -------------------------------------------------
        """
        right_match_number = len(self.match_records[self.match_pair_idx][2][:, 0:6])
        label1 = np.zeros(len(kpts0))
        if 1:
            c1 = dict(enumerate(epi_dist))
            f1 = zip(c1.values(), c1.keys())
            g1 = sorted(f1)
            k1 = np.array(g1)
            print("总的匹配个数:", len(k1))
            u1 = k1[k1[:, 0] < 0.01][: right_match_number]      # len(k) - 2000: len(k)        1e-4 0.001

            if len(u1) < right_match_number:
                print("点的个数不够了")
                u1 = k1[k1[:, 0] < 0.001][: right_match_number]      # 0.0001       0.01
            # print("38420849")
            # print(u)
            index1 = u1[:, 1]
            index1 = index1.astype(np.int32)
            label1[index1] = 1
        return label, u, label1, u1
        """
        # print("xixxixi", label, len(label), "0的个数", len(label)-np.count_nonzero(label))
        return label, u

    """
    def epi_detect_right_match(self, all_kpts_list):
        kpts0 = all_kpts_list[0]
        kpts1 = all_kpts_list[1]

        kpts0_complete = undist_points(kpts0, self.K0, self.dist0, self.ori_img_size0)
        kpts0 = np.stack([kpts0_complete[:, 2], kpts0_complete[:, 5]], axis=-1)

        kpts1_complete = undist_points(kpts1, self.K1, self.dist1, self.ori_img_size1)
        kpts1 = np.stack([kpts1_complete[:, 2], kpts1_complete[:, 5]], axis=-1)

        # validate epipolar geometry  验证对极几何
        e_mat = get_essential_mat(self.t0, self.t1, self.R0, self.R1)
        # print(888888)
        # print(kpts0.shape)
        epi_dist = get_epipolar_dist(kpts0, kpts1, self.K0, self.K1, self.ori_img_size0,
                                     self.ori_img_size1, e_mat)
        # print('max epipolar distance', np.max(epi_dist))
        label = np.ones(len(kpts0))

        number = len(self.match_records[self.match_pair_idx][2][:, 0:6])
        print("正确匹配的个数：", number)
        if 1:
            c = dict(enumerate(epi_dist))
            f = zip(c.values(), c.keys())
            g = sorted(f)
            k = np.array(g)
            print("总的匹配个数:", len(k))
            u = k[k[:, 0] < 0.001][: number]      # len(k) - 2000: len(k)        1e-4 0.001

            if len(u) < 2000:
                print("点的个数不够了")
                u = k[k[:, 0] < 0.01][: number]      # 0.0001       0.01
            # print("38420849")
            # print(u)
            index = u[:, 1]
            index = index.astype(np.int32)
            label[index] = 0

        # print("xixxixi", label, len(label), "0的个数", len(label)-np.count_nonzero(label))
        return label, u
    """

    def mask_detect_error_match(self, match_kpts):
        # visualize the mask file.
        mask_path = os.path.join(self.root, 'geolabel', 'mask.bin')
        mask_dict = read_mask(mask_path)
        mask = mask_dict.get(hash_int_pair(self.cidx0, self.cidx1))
        display, imgs_error_area_index, area_count = draw_mask(self.img0, self.img1, mask, downscale_ratio=1)
        label = np.ones(len(match_kpts[0]))
        for idx, val in enumerate(match_kpts):
            for idx1, val1 in enumerate(val):
                if imgs_error_area_index[idx][val1[1], val1[0]] == 1:
                    label[idx1] = 0
        # print(len(label), len(label)-np.count_nonzero(label))
        return label

    def extract_local_features(self, gray_list, n_sample=2000, peak_thld=0.04):
        """
        输入： 图像路径列表
        输出：
        3个列表
        每幅图像的sift detect检测出的kp返回值
        每幅图像的sift compute计算出的des描述子
        每幅图像的sift detect检测出的kp的6维旋转表示形式
        """
        cv_kpts_list = []
        sift_feat_list = []
        six_dimension_des_kpts_list = []

        model = get_model('loc_model')(None, **{'sift_desc': True,
                                                'n_sample': 2000,
                                                'peak_thld': peak_thld,
                                                'dense_desc': True,
                                                'upright': False})

        for idx, val in enumerate(gray_list):
            normalized_xy, cv_kpts, sift_desc, six_dimension_des_kpts = model.run_test_data(val)
            # raw_kpts = [np.array((i.pt[0], i.pt[1], i.size, i.angle, i.response)) for i in cv_kpts]
            # raw_kpts = np.stack(raw_kpts, axis=0)
            cv_kpts_list.append(cv_kpts)
            sift_feat_list.append(sift_desc)
            six_dimension_des_kpts_list.append(six_dimension_des_kpts)
        model.close()
        return cv_kpts_list, sift_feat_list, six_dimension_des_kpts_list

    def load_imgs(self, img_paths):
        rgb_list = []
        gray_list = []
        for idx, img_path in enumerate(img_paths):
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
            img = img[..., ::-1]
            rgb_list.append(img)
            gray_list.append(gray)
        return rgb_list, gray_list,

    def get_error_match(self, mode, display=True):
        img_paths = [self.img_path0, self.img_path1]
        rgb_list, gray_list = self.load_imgs(img_paths)
        cv_kpts_list, sift_feat_list, six_dimension_des_kpts_list = self.extract_local_features(gray_list)

        # print("匹配前总的点顺", len(six_dimension_des_kpts_list[0]), len(six_dimension_des_kpts_list[1]))
        if len(six_dimension_des_kpts_list[0]) <= 2500 or len(six_dimension_des_kpts_list[1]) <= 2500:
            cv_kpts_list, sift_feat_list, six_dimension_des_kpts_list = self.extract_local_features(gray_list,
                                                                                                    peak_thld=0.08)
            print("点的个数不够", len(six_dimension_des_kpts_list[0]), len(six_dimension_des_kpts_list[1]))

        matcher = MatcherWrapper()
        sift_match_index_relate_information, _, sift_match_kpts_non_normalized_xy_list, sift_match_six_dimension_des_kpts_list = matcher.get_matches(
            sift_feat_list[0], sift_feat_list[1], cv_kpts_list[0], cv_kpts_list[1], six_dimension_des_kpts_list[0],
            six_dimension_des_kpts_list[1],
            None, cross_check=False,
            err_thld=3, ransac=True, info='SIFT feautre')

        print("总的点数:", len(sift_match_six_dimension_des_kpts_list[0]))
        if mode == "good_dataset":
            label2, u = self.epi_detect_error_match(sift_match_six_dimension_des_kpts_list)
        elif mode == "bad_dataset":
            label1 = self.mask_detect_error_match(sift_match_kpts_non_normalized_xy_list)
            label2, u = self.epi_detect_error_match(sift_match_six_dimension_des_kpts_list)
            label2 = label1 * label2
        else:
            pass

        kpts0 = sift_match_six_dimension_des_kpts_list[0]
        kpts1 = sift_match_six_dimension_des_kpts_list[1]

        kpts0_complete = undist_points(kpts0, self.K0, self.dist0, self.ori_img_size0)
        kpts0 = np.stack([kpts0_complete[:, 2], kpts0_complete[:, 5]], axis=-1)

        kpts1_complete = undist_points(kpts1, self.K1, self.dist1, self.ori_img_size1)
        kpts1 = np.stack([kpts1_complete[:, 2], kpts1_complete[:, 5]], axis=-1)

        # print(kpts0.shape, kpts1.shape)
        kpts0_kpts1_4_non_select = np.column_stack((kpts0, kpts1))
        kpts0_kpts1_12_non_select = np.column_stack((kpts0_complete, kpts1_complete))

        # print("asdhkjahdsak")
        # print(kpts0_kpts1_4_non_select.shape, kpts0_kpts1_12_non_select.shape)

        error_select_label = label2
        error_select_label = error_select_label.astype(np.int32)
        # print(error_select_label)

        kpts0_kpts1_4 = kpts0_kpts1_4_non_select[error_select_label == 0]
        kpts0_kpts1_12 = kpts0_kpts1_12_non_select[error_select_label == 0]
        print("错误的匹配个数:", len(kpts0_kpts1_4))

        """
             筛选出正确的匹配
             # label2, u, label3, u1 = self.epi_detect_error_match(sift_match_six_dimension_des_kpts_list)
             right_select_label = label3
             right_select_label = right_select_label.astype(np.int32)

             kpts0_kpts1_4 = kpts0_kpts1_4_non_select[right_select_label == 1]
             kpts0_kpts1_12 = kpts0_kpts1_12_non_select[right_select_label == 1]

        """

        if display:
            print("错误索引:")
            print(u)

            temp = u[:, 1]
            b = sorted(enumerate(temp), key=lambda x: x[1])
            # print(b)
            c = [i[0] for i in b]
            d = np.array(range(len(b)))
            d = d.astype(np.int16)
            # print(d)
            e = np.zeros(len(b))
            e[c] = d
            # print("yyyyyyyyy")
            e = e.reshape((-1, 1))
            # print(e)
            u = np.concatenate((u, e), axis=1)
            # print(u)

            suoyin = []
            kpts0_kpts1_12_list = []
            kpts0_kpts1_12_list = np.array(kpts0_kpts1_12_list)
            for index, value in enumerate(u):
                suoyin.append(int(value[2]))
                kpts0_kpts1_12_list = np.append(kpts0_kpts1_12_list, kpts0_kpts1_12[int(value[2])])
                # kpts0_kpts1_12_list += kpts0_kpts1_12[int(value[2])]
                # hehe = kpts0_kpts1_12[int(value[2])]
                # print(hehe[0: 6].shape, hehe[6: ].shape)
                if index >= 20:
                    break

            kpts0_kpts1_12_list = kpts0_kpts1_12_list.reshape((-1, 12))
            kpts0 = np.stack([kpts0_kpts1_12_list[:, 2], kpts0_kpts1_12_list[:, 5]], axis=-1)
            kpts1 = np.stack([kpts0_kpts1_12_list[:, 8], kpts0_kpts1_12_list[:, 11]], axis=-1)
            # print(kpts0_kpts1_12_list)

            e_mat = get_essential_mat(self.t0, self.t1, self.R0, self.R1)
            epi_dist = get_epipolar_dist(kpts0, kpts1, self.K0, self.K1, self.ori_img_size0,
                                         self.ori_img_size1, e_mat)
            print(epi_dist)
            print(suoyin, len(suoyin))
            img_size0 = np.array((self.img0.shape[1], self.img0.shape[0]))
            img_size1 = np.array((self.img1.shape[1], self.img1.shape[0]))

            kpts0 = np.stack([kpts0_kpts1_12[:, 2], kpts0_kpts1_12[:, 5]], axis=-1)
            kpts1 = np.stack([kpts0_kpts1_12[:, 8], kpts0_kpts1_12[:, 11]], axis=-1)

            kpts0 = kpts0 * img_size0 / 2 + img_size0 / 2
            kpts1 = kpts1 * img_size1 / 2 + img_size1 / 2
            match_num = kpts0.shape[0]
            # match_idx = np.tile(np.array(range(0, match_num, 50))[..., None], [1, 2])
            match_idx = np.tile(np.array(suoyin)[..., None], [1, 2])  # range(0, 50)
            # bprint(match_idx, len(match_idx))
            display = draw_matches(self.img0, self.img1, kpts0, kpts1, match_idx, downscale_ratio=1.0)

            plt.xticks([])
            plt.yticks([])
            plt.imshow(display)
            plt.show()

        return kpts0_kpts1_4, kpts0_kpts1_12, rgb_list

    def get_all_match(self, mode, Display=False):
        x_right_4, x_right_12 = self.get_and_show_right_match()
        x_error_4, x_error_12, rgb_list = self.get_error_match(mode)
        # print("错误匹配个数:", len(x_error_4))
        # 从所有错误的匹配集合筛选出一部分错误匹配凑成2000个
        if 1:
            row_rand_array = np.arange(x_error_4.shape[0])
            np.random.shuffle(row_rand_array)

            # 一个生成4维 一个生成12维
            select_all_error_data = x_error_4[
                row_rand_array[0:2000 - len(x_right_4)]]
            # 生成2000个label和data
            select_all_data_4 = np.concatenate((x_right_4, select_all_error_data))

            # 生成12维
            select_all_error_data = x_error_12[
                row_rand_array[0:2000 - len(x_right_12)]]
            # 生成2000个label和data
            select_all_data_12 = np.concatenate((x_right_12, select_all_error_data))

            if Display:
                img_size0 = np.array((self.img0.shape[1], self.img0.shape[0]))
                img_size1 = np.array((self.img1.shape[1], self.img1.shape[0]))

                kpts0 = np.stack([select_all_error_data[:, 2], select_all_error_data[:, 5]], axis=-1)
                kpts1 = np.stack([select_all_error_data[:, 8], select_all_error_data[:, 11]], axis=-1)

                kpts0 = kpts0 * img_size0 / 2 + img_size0 / 2
                kpts1 = kpts1 * img_size1 / 2 + img_size1 / 2
                match_num = kpts0.shape[0]
                # match_idx = np.tile(np.array(range(0, match_num, 50))[..., None], [1, 2])
                match_idx = np.tile(np.array(range(0, 50))[..., None], [1, 2])
                display = draw_matches(self.img0, self.img1, kpts0, kpts1, match_idx, downscale_ratio=1.0)

                plt.xticks([])
                plt.yticks([])
                plt.imshow(display)
                plt.show()

            right_label = np.ones(len(x_right_4))
            error_label = np.zeros(len(x_error_4))
            label = np.concatenate((right_label, error_label))

            # 随机打乱
            row_rand_array = np.arange(select_all_data_4.shape[0])
            np.random.shuffle(row_rand_array)
            x_4 = select_all_data_4[row_rand_array]
            x_12 = select_all_data_12[row_rand_array]
            label = label[row_rand_array]

            print("总的匹配的个数:", len(x_4))
        return x_4, x_12, label, rgb_list

    def show_mask(self):
        # visualize the mask file.
        mask_path = os.path.join(self.root, 'geolabel', 'mask.bin')
        mask_dict = read_mask(mask_path)
        mask = mask_dict.get(hash_int_pair(self.cidx0, self.cidx1))
        if mask is not None:
            display, imgs_error_area_index, area_count = draw_mask(self.img0, self.img1, mask, downscale_ratio=1)
            return area_count, mask, display
        else:
            area_count = 0
            display = None
            return area_count, mask, display

"""
       电网数据集生成
"""
def datasets_generate_main(config):
    a = 0
    all_index = []
    for i in trange(300):   # 4501 9000
        visualize_new_class_def_inst = visualize_new_class_def("./Input_Data/57102be2877e1421026358af", i)

        right_match_count = len(visualize_new_class_def_inst.match_records[visualize_new_class_def_inst.match_pair_idx][2][:, 0:6])

        if visualize_new_class_def_inst.basename0 != a:
            print("bianhuan", i, i + 75)
            index_end = i + 75
            if a == 0:
                info = []
            else:
                # print(len(info), info)
                info1 = np.array(info)
                info1 = info1[np.lexsort(info1.T)]
                # print(len(info1), info1)
                all_index.append(info1[:, 0][:8].tolist())                              # 8
                all_index.append(info1[:, 0][len(info1) - 12:len(info1):3].tolist())    # len(info1) - 12:len(info1):3
                # print(all_index)
                info.clear()
            a = visualize_new_class_def_inst.basename0

        if i <= index_end:
            info.append([i, right_match_count])
        else:
            continue

        if len(all_index) >= 1:       # == 150
            select_index_result = [n for a in all_index for n in a]
            print(len(select_index_result), select_index_result)
            break


    # 显示一下图像的下标
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    for index, i in enumerate(select_index_result):
        visualize_new_class_def_inst = visualize_new_class_def("./Input_Data/57102be2877e1421026358af", i)
        print(visualize_new_class_def_inst.basename0, visualize_new_class_def_inst.basename1)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    # 一共如下几个信息   保存4个pkl
    # 1. 左右图像信息
    # 2. 四维坐标信息
    # 3. label标签信息
    # 4. 左右6维旋转信息
    # 5. 其他信息
    count = 0
    dataset = "good_dataset"

    data = {}
    data["img1s"] = []
    data["img2s"] = []
    data["xs_4"] = []
    data["label"] = []
    data["xs_12"] = []
    data["others"] = []

    cidx_list = []
    for index, i in enumerate(tqdm(select_index_result)):
        visualize_new_class_def_inst = visualize_new_class_def("./Input_Data/57102be2877e1421026358af", i)

        right_match_count = len(
            visualize_new_class_def_inst.match_records[visualize_new_class_def_inst.match_pair_idx][2][:, 0:6])

        x_4, x_12, label, rgb_list = visualize_new_class_def_inst.get_all_match("good_dataset")

        print(x_4.shape, right_match_count)
        data["img1s"].append(rgb_list[0])
        data["img2s"].append(rgb_list[1])
        data["xs_4"].append([x_4])
        data["label"].append(label)
        data["xs_12"].append(x_12)
        data["others"].append([visualize_new_class_def_inst.cidx0, visualize_new_class_def_inst.cidx1,
                               visualize_new_class_def_inst.match_num])

        count += 1
        if count == 904:  # 900 904
            print("结束")
            # train_data_path = "E:/NM-Net-Initial/datasets/COLMAP_hehe8/train"
            # valid_data_path = "E:/NM-Net-Initial/datasets/COLMAP_hehe8/valid"
            # test_data_path = "E:/NM-Net-Initial/datasets/COLMAP_hehe8/test"
            train_data_path = os.path.join(config.datasets_output_path, "/COLMAP_hehe7/train")
            valid_data_path = os.path.join(config.datasets_output_path, "/COLMAP_hehe7/valid")
            test_data_path = os.path.join(config.datasets_output_path, "/COLMAP_hehe7/test")

            if not os.path.isdir(train_data_path):
                os.makedirs(train_data_path)
            if not os.path.isdir(valid_data_path):
                os.makedirs(valid_data_path)
            if not os.path.isdir(test_data_path):
                os.makedirs(test_data_path)

            if 1:
                var_name_list = ["img1s", "img2s", "xs_4", "label", "xs_12", "others"]
                data_temp = {}

                length = len(data["img1s"])
                data_temp["img1s"] = data["img1s"][:int(0.7 * length)]
                data_temp["img2s"] = data["img2s"][:int(0.7 * length)]
                data_temp["xs_4"] = data["xs_4"][:int(0.7 * length)]
                data_temp["label"] = data["label"][:int(0.7 * length)]
                data_temp["xs_12"] = data["xs_12"][:int(0.7 * length)]
                data_temp["others"] = data["others"][:int(0.7 * length)]
                print("训练集长度: ", len(data_temp["img1s"]))
                for var_name in var_name_list:
                    in_file_name = os.path.join(train_data_path, var_name) + ".pkl"
                    visualize_new_class_def_inst.save_pkl(in_file_name, data_temp[var_name])

                data_temp = {}
                data_temp["img1s"] = data["img1s"][int(0.7 * length): int(0.85 * length)]
                data_temp["img2s"] = data["img2s"][int(0.7 * length): int(0.85 * length)]
                data_temp["xs_4"] = data["xs_4"][int(0.7 * length): int(0.85 * length)]
                data_temp["label"] = data["label"][int(0.7 * length): int(0.85 * length)]
                data_temp["xs_12"] = data["xs_12"][int(0.7 * length): int(0.85 * length)]
                data_temp["others"] = data["others"][int(0.7 * length): int(0.85 * length)]
                print("验证集长度: ", len(data_temp["img1s"]))
                for var_name in var_name_list:
                    in_file_name = os.path.join(valid_data_path, var_name) + ".pkl"
                    visualize_new_class_def_inst.save_pkl(in_file_name, data_temp[var_name])

                data_temp = {}
                data_temp["img1s"] = data["img1s"][int(0.85 * length): length]
                data_temp["img2s"] = data["img2s"][int(0.85 * length): length]
                data_temp["xs_4"] = data["xs_4"][int(0.85 * length): length]
                data_temp["label"] = data["label"][int(0.85 * length): length]
                data_temp["xs_12"] = data["xs_12"][int(0.85 * length): length]
                data_temp["others"] = data["others"][int(0.85 * length): length]
                print("测试集长度: ", len(data_temp["img1s"]))
                for var_name in var_name_list:
                    in_file_name = os.path.join(test_data_path, var_name) + ".pkl"
                    visualize_new_class_def_inst.save_pkl(in_file_name, data_temp[var_name])
            break

