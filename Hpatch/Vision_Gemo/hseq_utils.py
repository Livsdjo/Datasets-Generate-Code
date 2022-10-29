#!/usr/bin/env python3
"""
Copyright 2017, Zixin Luo, HKUST.
HSequence evaluation tools.
"""

import os
import glob
import pickle
import random
import cv2
import numpy as np

from utils.opencvhelper import SiftWrapper


class HSeqData(object):
    def __init__(self):
        self.img = []
        self.patch = []
        self.kpt_param = []
        self.coord = []
        self.homo = []
        self.img_feat = []
        self.desc = []


class HSeqUtils(object):
    def __init__(self, config):
        self.seqs = []
        self.seq_i_num = 0
        self.seq_v_num = 0

        seq_types = ['%s_*' % i for i in config['seq']]
        for files in seq_types:
            tmp_seqs = glob.glob(os.path.join(config['root'], files))
            tmp_seqs.sort()
            if files[0] == 'i':
                self.seq_i_num = len(tmp_seqs)
            if files[0] == 'v':
                self.seq_v_num = len(tmp_seqs)
            self.seqs.extend(tmp_seqs)
        self.seqs = self.seqs[config['start_idx']:]
        self.seq_num = len(self.seqs)
        self.suffix = config['suffix']
        # for detector config
        self.upright = config['upright']
        # for data parsing
        self.sample_num = 6000   # config['kpt_n']
        self.patch_scale = 3               # 6
        # for data des
        self.sift_desc = config['sift_desc']


    """
        这个是contextdesc测试中读取hpatch的代码
    """
    def get_data(self, seq_idx, dense_desc):
        random.seed(0)
        if self.suffix is None:
            sift_wrapper = SiftWrapper(n_feature=self.sample_num, peak_thld=0.04)
            sift_wrapper.ori_off = self.upright
            sift_wrapper.create()

        hseq_data = HSeqData()
        seq_name = self.seqs[seq_idx]

        for img_idx in range(1, 7):
            # read image features.
            img_feat = np.load(os.path.join(seq_name, '%d_img_feat.npy' % img_idx))
            # read images.
            print(os.path.join(seq_name, '%d.ppm' % img_idx))
            # path = "E:/contextdesc-master/contextdesc-master/hpatches-sequences-release/hpatches-sequences-release/i_dc/1.ppm"
            # img =  cv2.imread(path)
            img = cv2.imread(os.path.join(seq_name, '%d.ppm' % img_idx))
            # print(os.path.join(seq_name, '%d.ppm' % img_idx))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_size = img.shape

            npy_kpts, cv_kpts = sift_wrapper.detect(gray)
            if len(cv_kpts) < 6000:
                # sift_wrapper.peak_thld = 0.0067
                sift_wrapper.peak_thld = 0.0010
                sift_wrapper.create()
                npy_kpts, cv_kpts = sift_wrapper.detect(gray)
                print("个数:", len(cv_kpts))
            else:
                print("个数：", len(cv_kpts))

            if self.suffix is None:
                # print("*********************")
                npy_kpts, cv_kpts = sift_wrapper.detect(gray)
                if not dense_desc:
                    sift_wrapper.build_pyramid(gray)
                    patches = sift_wrapper.get_patches(cv_kpts)
                else:
                    patches = None
            else:
                with open(os.path.join(seq_name, ('%d' + self.suffix + '.pkl') % img_idx), 'rb') as handle:
                    data_dict = pickle.load(handle, encoding='latin1')
                npy_kpts = data_dict['npy_kpts']
                if not dense_desc:
                    patches = data_dict['patches']
                else:
                    patches = None

            if self.sift_desc:
                sift_desc1 = sift_wrapper.compute(gray, cv_kpts)
            else:
                sift_desc1 = None

            kpt_num = npy_kpts.shape[0]

            # compose affine crop matrix.
            crop_mat = np.zeros((kpt_num, 6))
            # rely on the SIFT orientation estimation.
            m_cos = np.cos(-npy_kpts[:, 3]) * self.patch_scale * npy_kpts[:, 2]
            m_sin = np.sin(-npy_kpts[:, 3]) * self.patch_scale * npy_kpts[:, 2]
            # m_cos = self.patch_scale * npy_kpts[:, 2]
            # m_sin = self.patch_scale * npy_kpts[:, 2]

            crop_mat[:, 0] = m_cos / float(img_size[1])
            crop_mat[:, 1] = m_sin / float(img_size[1])
            crop_mat[:, 2] = (npy_kpts[:, 0] - img_size[1] / 2.) / (img_size[1] / 2.)
            crop_mat[:, 3] = -m_sin / float(img_size[0])
            crop_mat[:, 4] = m_cos / float(img_size[0])
            crop_mat[:, 5] = (npy_kpts[:, 1] - img_size[0] / 2.) / (img_size[0] / 2.)
            npy_kpts = npy_kpts[:, 0:2]

            # read homography matrix.
            if img_idx > 1:
                homo_mat = open(os.path.join(seq_name, 'H_1_%d' % img_idx)).read().splitlines()
                homo_mat = np.array([float(i) for i in ' '.join(homo_mat).split()])
                homo_mat = np.reshape(homo_mat, (3, 3))
            else:
                homo_mat = None

            hseq_data.img.append(img)
            hseq_data.kpt_param.append(crop_mat)
            hseq_data.patch.append(patches)
            hseq_data.coord.append(npy_kpts)
            hseq_data.homo.append(homo_mat)
            hseq_data.img_feat.append(img_feat)
            hseq_data.desc.append(sift_desc1)

        return seq_name, hseq_data


    """
        这个是自己写的读取hpatch数据集的代码
        读取的是原版haptch数据集的文件 
    """
    def get_data_yuanban(self, seq_idx, dense_desc):
        random.seed(0)
        if self.suffix is None:
            sift_wrapper = SiftWrapper(n_feature=self.sample_num, peak_thld=0.04)
            sift_wrapper.ori_off = self.upright
            sift_wrapper.create()

        hseq_data = HSeqData()
        seq_name = self.seqs[seq_idx]


        for img_idx in range(1, 7):
            # read image features.
            # img_feat = np.load(os.path.join(seq_name, '%d_img_feat.npy' % img_idx))
            # read images.
            print(os.path.join(seq_name, '%d.ppm' % img_idx))
            # path = "E:/contextdesc-master/contextdesc-master/hpatches-sequences-release/hpatches-sequences-release/i_dc/1.ppm"
            # img =  cv2.imread(path)
            img = cv2.imread(os.path.join(seq_name, '%d.ppm' % img_idx))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_size = img.shape

            npy_kpts, cv_kpts = sift_wrapper.detect(gray)
            if len(cv_kpts) < 2000:
                sift_wrapper.peak_thld = 0.0067
                sift_wrapper.create()
                npy_kpts, cv_kpts = sift_wrapper.detect(gray)
                print("个数:", len(cv_kpts))
            else:
                print("个数：", len(cv_kpts))

            if self.suffix is None:
                # print("*********************")
                npy_kpts, cv_kpts = sift_wrapper.detect(gray)
                if not dense_desc:
                    sift_wrapper.build_pyramid(gray)
                    patches = sift_wrapper.get_patches(cv_kpts)
                else:
                    patches = None
            else:
                print("35749375389738")
                with open(os.path.join(seq_name, ('%d' + self.suffix + '.pkl') % img_idx), 'rb') as handle:
                    data_dict = pickle.load(handle, encoding='latin1')
                npy_kpts = data_dict['npy_kpts']
                if not dense_desc:
                    patches = data_dict['patches']
                else:
                    patches = None

            if self.sift_desc:
                sift_desc1 = sift_wrapper.compute(gray, cv_kpts)
            else:
                sift_desc1 = None

            kpt_num = npy_kpts.shape[0]

            # compose affine crop matrix.
            crop_mat = np.zeros((kpt_num, 6))
            # rely on the SIFT orientation estimation.
            m_cos = np.cos(-npy_kpts[:, 3]) * self.patch_scale * npy_kpts[:, 2]
            m_sin = np.sin(-npy_kpts[:, 3]) * self.patch_scale * npy_kpts[:, 2]
            # m_cos = self.patch_scale * npy_kpts[:, 2]
            # m_sin = self.patch_scale * npy_kpts[:, 2]

            crop_mat[:, 0] = m_cos / float(img_size[1])
            crop_mat[:, 1] = m_sin / float(img_size[1])
            crop_mat[:, 2] = (npy_kpts[:, 0] - img_size[1] / 2.) / (img_size[1] / 2.)
            crop_mat[:, 3] = -m_sin / float(img_size[0])
            crop_mat[:, 4] = m_cos / float(img_size[0])
            crop_mat[:, 5] = (npy_kpts[:, 1] - img_size[0] / 2.) / (img_size[0] / 2.)
            npy_kpts = npy_kpts[:, 0:2]

            # read homography matrix.
            if img_idx > 1:
                homo_mat = open(os.path.join(seq_name, 'H_1_%d' % img_idx)).read().splitlines()
                homo_mat = np.array([float(i) for i in ' '.join(homo_mat).split()])
                homo_mat = np.reshape(homo_mat, (3, 3))
            else:
                homo_mat = None

            # print(666666666, npy_kpts.shape, npy_kpts)
            hseq_data.img.append(img)
            hseq_data.kpt_param.append(crop_mat)
            hseq_data.patch.append(patches)
            hseq_data.coord.append(npy_kpts)
            hseq_data.homo.append(homo_mat)
            hseq_data.img_feat.append(img_feat)
            hseq_data.desc.append(sift_desc1)

        return seq_name, hseq_data



