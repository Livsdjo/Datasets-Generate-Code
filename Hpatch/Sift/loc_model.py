import sys
from queue import Queue
from threading import Thread

import os
from struct import unpack
import numpy as np
import cv2
import tensorflow as tf

from .base_model import BaseModel
from .cnn_wrapper.descnet import GeoDesc, DenseGeoDesc
from .cnn_wrapper.augdesc import MatchabilityPrediction

sys.path.append('..')

from utils.opencvhelper import SiftWrapper
from utils.spatial_transformer import transformer_crop

import numpy as np
np.set_printoptions(suppress=True)


class LocModel(BaseModel):
    output_tensors = ["conv6_feat:0", "kpt_mb:0"]
    default_config = {'n_feature': 0, "n_sample": 0,
                      'batch_size': 512, 'sift_wrapper': None, 'upright': False, 'scale_diff': False,
                      'dense_desc': False, 'sift_desc': True, 'peak_thld': 0.0067, 'edge_thld': 10, 'max_dim': 1280}
                                                                # 'peak_thld': 0.0067, 'edge_thld': 30  10  0.0128
    def _init_model(self):
        self.sift_wrapper = SiftWrapper(
            n_feature=self.config['n_feature'],
            n_sample=self.config['n_sample'],
            peak_thld=self.config['peak_thld'],
            edge_thld=self.config['edge_thld']
            )
        self.sift_wrapper.standardize = False  # the network has handled this step.
        self.sift_wrapper.ori_off = self.config['upright']
        self.sift_wrapper.pyr_off = not self.config['scale_diff']
        self.sift_wrapper.create()

    def _run(self, data, **kwargs):
        def _worker(patch_queue, sess, loc_feat, kpt_mb):
            """The worker thread."""
            while True:
                patch_data = patch_queue.get()
                if patch_data is None:
                    return
                loc_returns = sess.run(self.output_tensors,
                                       feed_dict={"input:0": np.expand_dims(patch_data, -1)})
                loc_feat.append(loc_returns[0])
                kpt_mb.append(loc_returns[1])
                patch_queue.task_done()
        gray_img = np.squeeze(data, axis=-1).astype(np.uint8)
        # detect SIFT keypoints.
        npy_kpts, cv_kpts = self.sift_wrapper.detect(gray_img)

        if self.config['sift_desc']:
            sift_desc = self.sift_wrapper.compute(gray_img, cv_kpts)
        else:
            sift_desc = None

        kpt_xy = np.stack(((npy_kpts[:, 0] - gray_img.shape[1] / 2.) / (gray_img.shape[1] / 2.),
                           (npy_kpts[:, 1] - gray_img.shape[0] / 2.) / (gray_img.shape[0] / 2.)),
                          axis=-1)

        num_patch = len(cv_kpts)

        if not self.config['dense_desc']:
            self.sift_wrapper.build_pyramid(gray_img)
            all_patches = self.sift_wrapper.get_patches(cv_kpts)
            # get iteration number
            batch_size = self.config['batch_size']
            if num_patch % batch_size > 0:
                loop_num = int(np.floor(float(num_patch) / float(batch_size)))
            else:
                loop_num = int(num_patch / batch_size - 1)
            # create input thread
            loc_feat = []
            kpt_mb = []
            patch_queue = Queue()
            worker_thread = Thread(target=_worker, args=(patch_queue, self.sess, loc_feat, kpt_mb))
            worker_thread.daemon = True
            worker_thread.start()
            # enqueue
            for i in range(loop_num + 1):
                if i < loop_num:
                    patch_queue.put(all_patches[i * batch_size: (i + 1) * batch_size])
                else:
                    patch_queue.put(all_patches[i * batch_size:])
            # poison pill
            patch_queue.put(None)
            # wait for extraction.
            worker_thread.join()
            loc_feat = np.concatenate(loc_feat, axis=0)
            kpt_mb = np.concatenate(kpt_mb, axis=0)
        else:
            # import cv2
            # compose affine crop matrix.
            patch_scale = 3                         # 6
            patch_param = np.zeros((num_patch, 6))

            # m_cos = np.cos(npy_kpts[:, 3]) * patch_scale * npy_kpts[:, 2]
            # m_sin = np.sin(npy_kpts[:, 3]) * patch_scale * npy_kpts[:, 2]

            m_cos = patch_scale * npy_kpts[:, 2]
            m_sin = patch_scale * npy_kpts[:, 2]

            short_side = float(min(gray_img.shape[0], gray_img.shape[1]))

            patch_param[:, 0] = m_cos / short_side
            patch_param[:, 1] = m_sin / short_side
            patch_param[:, 2] = kpt_xy[:, 0]
            patch_param[:, 3] = -m_sin / short_side
            patch_param[:, 4] = m_cos / short_side
            patch_param[:, 5] = kpt_xy[:, 1]

            max_dim = max(gray_img.shape[0], gray_img.shape[1])
            if max_dim > self.config['max_dim']:
                downsample_ratio = self.config['max_dim'] / float(max_dim)
                gray_img = cv2.resize(gray_img, (0, 0), fx=downsample_ratio, fy=downsample_ratio)

            gray_img = gray_img[..., np.newaxis]

            if 0:
                input_dict = {"input/img:0": np.expand_dims(gray_img, 0),
                              "input/kpt_param:0": np.expand_dims(patch_param, 0)}
                local_returns = self.sess.run(self.output_tensors, feed_dict=input_dict)
                loc_feat = local_returns[0]
                kpt_mb = local_returns[1]
            else:
                pass

        if 0:
            return loc_feat, kpt_mb, kpt_xy, cv_kpts, sift_desc
        else:
            return kpt_xy, cv_kpts, sift_desc, patch_param

    def _construct_network(self):
        """Model for patch description."""

        if self.config['dense_desc']:
            with tf.name_scope('input'):
                ph_imgs = tf.placeholder(dtype=tf.float32, shape=(
                    None, None, None, 1), name='img')
                ph_kpt_params = tf.placeholder(tf.float32, shape=(None, None, 6), name='kpt_param')
            kpt_xy = tf.concat((ph_kpt_params[:, :, 2, None], ph_kpt_params[:, :, 5, None]), axis=-1)
            kpt_theta = tf.reshape(ph_kpt_params, (tf.shape(ph_kpt_params)[0], tf.shape(ph_kpt_params)[1], 2, 3))
            mean, variance = tf.nn.moments(
                tf.cast(ph_imgs, tf.float32), axes=[1, 2], keep_dims=True)
            norm_input = tf.nn.batch_normalization(ph_imgs, mean, variance, None, None, 1e-5)
            config_dict = {}
            config_dict['pert_theta'] = kpt_theta
            config_dict['patch_sampler'] = transformer_crop
            tower = DenseGeoDesc({'data': norm_input, 'kpt_coord': kpt_xy},
                          is_training=False, resue=False, **config_dict)
        else:
            input_size = (32, 32)
            patches = tf.placeholder(
                dtype=tf.float32, shape=(None, input_size[0], input_size[1], 1), name='input')
            # patch standardization
            mean, variance = tf.nn.moments(
                tf.cast(patches, tf.float32), axes=[1, 2], keep_dims=True)
            patches = tf.nn.batch_normalization(patches, mean, variance, None, None, 1e-5)
            tower = GeoDesc({'data': patches}, is_training=False, reuse=False)

        conv6_feat = tower.get_output_by_name('conv6')
        conv6_feat = tf.squeeze(conv6_feat, axis=[1, 2], name='conv6_feat')

        with tf.compat.v1.variable_scope('kpt_m'):
            inter_feat = tower.get_output_by_name('conv5')
            matchability_tower = MatchabilityPrediction(
                {'data': inter_feat}, is_training=False, reuse=False)
            mb = matchability_tower.get_output()
        mb = tf.squeeze(mb, axis=[1, 2], name='kpt_mb')























# 不要的
"""
        # 列表生成式
        all_detect_key_point = [[int(value.pt[0]), int(value.pt[1])] for value in cv_kpts]
        print(cv_kpts[0].pt, len(all_detect_key_point))
        print(all_detect_key_point)
        for value in cv_kpts:
         print(value.pt)
        ##保存
        
        fileObject = open('sampleList.txt', 'w')
        for ip in cv_kpts:
            fileObject.write(str(ip.pt))
            fileObject.write('\n')
        fileObject.close()
        
        root = 'E:/data/000000000000000000000000/'
        basename0 = "00000020"
        basename1 = "00000033"
        img_path0 = os.path.join(root, 'undist_images', basename0 + '.jpg')
        img_path1 = os.path.join(root, 'undist_images', basename1 + '.jpg')
        img0 = cv2.imread(img_path0)[..., ::-1]
        img1 = cv2.imread(img_path1)[..., ::-1]
        cam_path = os.path.join(root, 'geolabel', 'cameras.txt')
        cam_dict = read_cams(cam_path)
        cam0, cam1 = cam_dict[20], cam_dict[33]
        K0, K1 = cam0[0], cam1[0]
        p0f0 = [(K0[0, 2], K0[1, 2]), (K0[0, 0], K0[1, 1])]
        p1f1 = [(K1[0, 2], K1[1, 2]), (K1[0, 0], K1[1, 1])]
        t0, t1 = cam0[1], cam1[1]
        R0, R1 = cam0[2], cam1[2]
        dist0, dist1 = cam0[3], cam1[3]
        ori_img_size0, ori_img_size1 = cam0[4], cam1[4]
        # visualize the keypoint file.
        kpt_path0 = os.path.join(root, 'img_kpts', basename0 + '.bin')
        kpt_path1 = os.path.join(root, 'img_kpts', basename1 + '.bin')
        # parse keypoint file.
        kpts0, kpts1 = read_kpt(kpt_path0), read_kpt(kpt_path1)
        print("坐标")
        print(len(kpts0))
        # undistortion.
        # kpts0 = undist_points(kpts0, K0, dist0, ori_img_size0)
        # kpts1 = undist_points(kpts1, K1, dist1, ori_img_size1)
        # extract normalized coordinates.
        kpts0 = np.stack([kpts0[:, 2], kpts0[:, 5]], axis=-1)
        kpts1 = np.stack([kpts1[:, 2], kpts1[:, 5]], axis=-1)
        # get image coordinates.
        img_size0 = np.array((img0.shape[1], img0.shape[0]))
        img_size1 = np.array((img1.shape[1], img1.shape[0]))
        kpts0 = kpts0 * img_size0 / 2 + img_size0 / 2
        kpts1 = kpts1 * img_size1 / 2 + img_size1 / 2
        print(type(kpts0))

        kpts0 = kpts0.astype(np.int32).tolist()
        print(kpts0)
        count = 0
        for value in kpts0:
            if value in all_detect_key_point:
                count += 1
        print(count)

        for value in kpts0:
            print(value)
"""







