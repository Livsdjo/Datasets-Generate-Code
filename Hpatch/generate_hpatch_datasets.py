import os

from queue import Queue
from threading import Thread

import yaml
import cv2
import numpy as np
import progressbar

import tensorflow as tf

from Vision_Gemo.hseq_utils import HSeqUtils
from Vision_Gemo.evaluator import Evaluator
from Sift import get_model
from Sift.inference_model import inference
from tqdm import trange
import pickle
import matplotlib.pyplot as plt


FLAGS = tf.app.flags.FLAGS

# Params for hpatches benchmark
tf.app.flags.DEFINE_string('config', None, """Path to the configuration file.""")

def recoverer(sess, model_path, meta_graph_path=None):
    """
    Recovery parameters from a pretrained model.
    Args:
        sess: The tensorflow session instance.
        model_path: Checkpoint file path.
    Returns:
        Nothing
    """
    if meta_graph_path is None:
        restore_var = tf.compat.v1.global_variables()
        restorer = tf.compat.v1.train.Saver(restore_var)
    else:
        restorer = tf.train.import_meta_graph(meta_graph_path)
    restorer.restore(sess, model_path)

"""
      
"""
def loader(hseq_utils, dense_desc, producer_queue):
    print(2222222, hseq_utils.seq_num)
    # hseq_utils.seq_num
    for seq_idx in trange(116):         # hseq_utils.seq_num
        if seq_idx == 71:       # i_dc
            continue
        seq_name, hseq_data = hseq_utils.get_data(seq_idx, dense_desc)
        # print()

        for i in range(6):
            gt_homo = [seq_idx, seq_name] if i == 0 else hseq_data.homo[i]
            # print(99999999, gt_homo)
            # print("ehehhesk", gt_homo, hseq_data.img_feat[i].shape)
            producer_queue.put([hseq_data.img[i],
                                hseq_data.kpt_param[i],
                                None,
                                None, # hseq_data.img_feat[i],
                                hseq_data.coord[i],
                                hseq_data.desc[i],
                                gt_homo])
        # print(2222222, producer_queue)
    producer_queue.put(None)
    print(333333333, producer_queue)



"""
    这个地方想用与主函数了
    一共6个部分
    img1
    img2
    x_4
    x_12
    others    图像的下标指引 以及图像正确标签的个数
    label


"""
def matcher(producer_queue, sess, evaluator, diaplay=True):
    record = []
    count = 0

    # 作为总的数据集容器数据库
    data = {}
    data["img1s"] = []
    data["img2s"] = []
    data["xs_4"] = []
    data["label"] = []
    data["xs_12"] = []
    data["others"] = []
    while True:
        queue_data = producer_queue.get()    # 相当于数据队列
        if queue_data is None:
            return data
        record.append(queue_data)
        if len(record) < 6:
            continue
        count += 1

        """
             一般来说就是6副图像作为了一组放在一个record里边
             在这里就是把record[0]和其他5个record进行想用的匹配并生成标签
             或者说能做其他的匹配这样的就可以增加数据集的数量了
        
        """
        img1, kpt_param1, patch1, img_feat1, ref_coord1, ref_feat1, seq_info1 = record[0]
        # seq_idx = seq_info1[0]
        seq_name = os.path.basename(seq_info1[1])

        recall = 0

        for i in range(1, 6):
            img2, kpt_param2, patch2, img_feat2, test_coord2, test_feat2, gt_homo2 = record[i]
            # print(55555555, ref_feat1.shape, test_feat2.shape, ref_coord1.shape, test_coord2.shape)
            putative_matches = evaluator.feature_matcher(sess, ref_feat1, test_feat2)
            inlier_matches = evaluator.get_inlier_matches(
                ref_coord1, test_coord2, putative_matches, gt_homo2)
            # Calculate recall
            num_inlier = len(inlier_matches)
            gt_num, gt_mask = evaluator.get_gt_matches(ref_coord1, test_coord2, gt_homo2)
            recall += (num_inlier / max(gt_num, 1)) / 5

            """
                没有归一化的坐标 只不过要用来算label罢了
            """
            kpts1 = np.array([ref_coord1[m.queryIdx] for m in putative_matches])
            kpts2 = np.array([test_coord2[m.trainIdx] for m in putative_matches])
            """
                6维描述坐标
            """
            six_dim_kpts1 = np.array([kpt_param1[m.queryIdx] for m in putative_matches])
            six_dim_kpts2 = np.array([kpt_param2[m.trainIdx] for m in putative_matches])
            # print("esuweyrwo", six_dim_kpts1.shape)
            """
                归一化坐标 6维坐标拼接
            """
            """
            six_dim_kpts1[2],  six_dim_kpts1[5]
            six_dim_kpts2[2],  six_dim_kpts2[5]
            """
            kpts_normal_0 = np.stack([six_dim_kpts1[:, 2], six_dim_kpts1[:, 5]], axis=-1)
            kpts_normal_1 = np.stack([six_dim_kpts2[:, 2], six_dim_kpts2[:, 5]], axis=-1)

            # print(kpts0.shape, kpts1.shape)
            xs_4 = np.column_stack((kpts_normal_0, kpts_normal_1))
            xs_12 = np.column_stack((six_dim_kpts1, six_dim_kpts2))


            _, mask = cv2.findFundamentalMat(
                kpts1[gt_mask, :], kpts2[gt_mask, :], cv2.RANSAC, 4)

            # print("标签的结果")
            # print(len(gt_mask), gt_mask)
            # print(mask)
            mask = mask.reshape(-1).astype(np.bool)
            # print(len(mask), mask)
            """
                将两个标签聚合在一起
            """
            label_temp = np.array([0] * 2000)
            index_temp = []
            for index, value in enumerate(gt_mask):
                if value == True:
                    label_temp[index] = 1
                    index_temp.append(index)
                else:
                    pass
            for index, value in enumerate(mask):
                if value == False:
                    label_temp[index_temp[index]] = 0
            # 最终所认定的正确匹配的个数
            match_num = len(mask)
            # ("dhkjadha", len(label_temp))

            # print(gt_homo2)


            """
                加入相关数据
            """
            data["img1s"].append(img1)
            data["img2s"].append(img2)
            data["label"].append(label_temp)
            data["others"].append([seq_name + "_" + "0", seq_name + "_" + str(i), match_num, img1.shape, img2.shape])
            data["xs_4"].append([xs_4])
            data["xs_12"].append([xs_12])
            # print(7777777, seq_name + str(i))


        record = []

        print("数据集名称", seq_name)



def prepare_reg_feat(hseq_utils, reg_model, overwrite):
    in_img_path = []
    out_img_feat_list = []
    # print(hseq_utils.seqs)
    print(111111111)
    v_count = 0
    i_count = 0
    for seq_name in hseq_utils.seqs:
        temp = seq_name.split('/')[-1]
        temp = temp.split("\\")[-1]
        temp = temp.split('_')[0]
        if temp == 'v':
            v_count += 1
        else:
            i_count += 1
        # print(seq_name.split('/'))
        for img_idx in range(1, 7):
            img_feat_path = os.path.join(seq_name, '%d_img_feat.npy' % img_idx)
            # print(111111, img_feat_path)
            if not os.path.exists(img_feat_path) or overwrite:
                in_img_path.append(os.path.join(seq_name, '%d.ppm' % img_idx))
                out_img_feat_list.append(img_feat_path)
    print(v_count, i_count)

    if len(in_img_path) > 0:
        model = get_model('reg_model')(reg_model)
        prog_bar = progressbar.ProgressBar()
        prog_bar.max_value = len(in_img_path)
        for idx, val in enumerate(in_img_path):
            img = cv2.imread(val)
            img = img[..., ::-1]
            reg_feat = model.run_test_data(img)
            # print("hseq_eval", reg_feat)
            np.save(out_img_feat_list[idx], reg_feat)
            prog_bar.update(idx)
        model.close()


def save_pkl(data_path, data):
    pickle_file = open(data_path, 'wb')  # 创建一个pickle文件，文件后缀名随意,但是打开方式必须是wb（以二进制形式写入）
    pickle.dump(data, pickle_file)  # 将列表倒入文件
    pickle_file.close()


def hseq_eval(config1):
    # FLAGS.config = "E:/contextdesc-master\contextdesc-master\configs/hseq_eval.yaml"
    path = "./hseq_eval.yaml"
    with open(path, 'r') as f:
        test_config = yaml.load(f, Loader=yaml.FullLoader)
    # Configure dataset
    hseq_utils = HSeqUtils(test_config['hseq'])
    prepare_reg_feat(hseq_utils, test_config['eval']['reg_model'], test_config['hseq']['overwrite'])
    # Configure evaluation
    evaluator = Evaluator(test_config['eval'])
    # Construct inference networks.
    output_tensors = inference(test_config['network'])
    # Create the initializier.
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        print(11111111)
        recoverer(sess, test_config['eval']['loc_model'])

        producer_queue = Queue(maxsize=696)  # 18   # 116 * 6   180
        loader(hseq_utils, test_config['network']['dense_desc'], producer_queue)
        # print(producer_queue)
        # print(66666666)
        # 获得全部的数据
        data = matcher(producer_queue, sess, evaluator)
        # print(555555555)
        # 保存数据
        # train_data_path = "E:/contextdesc-master/contextdesc-master/hpatches-sequences-release\Hpatch_Data/train"  # "E:/NM-net-xiexie/datasets/Hpatch_Data/train"
        # valid_data_path = "E:/contextdesc-master/contextdesc-master/hpatches-sequences-release\Hpatch_Data/valid"  # "E:/NM-net-xiexie/datasets/Hpatch_Data/valid"
        # test_data_path = "E:/contextdesc-master/contextdesc-master/hpatches-sequences-release\Hpatch_Data/test"   # "E:/NM-net-xiexie/datasets/Hpatch_Data/test"
        train_data_path = os.path.join(config1.datasets_output_path, "/COLMAP_heheA/train")
        valid_data_path = os.path.join(config1.datasets_output_path, "/COLMAP_heheA/valid")
        test_data_path = os.path.join(config1.datasets_output_path, "/COLMAP_heheA/test")

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
                save_pkl(in_file_name, data_temp[var_name])

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
                save_pkl(in_file_name, data_temp[var_name])

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
                save_pkl(in_file_name, data_temp[var_name])

    print(evaluator.stats)

from config import get_config, print_usage
config, unparsed = get_config()
def main(argv=None):  # pylint: disable=unused-argument
    """Program entrance."""
    # tf.flags.mark_flags_as_required(['config'])
    config, unparsed = get_config()
    hseq_eval(config)
    print("Finish !!!")



if __name__ == '__main__':
    main()
