import pickle
import os
import numpy as np
import random
from tqdm import trange


def data_merge(config):
    # Now load data.
    var_name_list = [
        "xs_4", "label", "img1s", "img2s", "xs_12", "others"
    ]

    data = {}
    # data_folder = "E:/NM-Net-Initial/datasets/COLMAP_good_100"
    # data_folder = "E:/NM-Net-Initial/datasets/COLMAP_go0d"
    # data_folder = "E:/NM-Net-Initial/datasets/COLMAP_900_good"
    # data_folder = "E:/NM-Net-Initial/datasets/COLMAP2"

    # data_folder = "E:/NM-Net-xiexie/datasets/COLMAP_5a533e8034d7582116e34209"
    # data_folder = "E:/NM-Net-Initial/datasets/COLMAP_hehe7"
    data_folder = os.path.join(config.datasets_output_path, "COLMAP_hehe7")

    mode_list = ["train", "valid", "test"]
    for var_mode in mode_list:
        data_path = os.path.join(data_folder, var_mode)
        print("路径", data_path)

        for var_name in var_name_list:
            in_file_name = os.path.join(data_path, var_name) + ".pkl"
            with open(in_file_name, "rb") as ifp:
                if var_name in data:
                    data[var_name] += pickle.load(ifp)
                else:
                    data[var_name] = pickle.load(ifp)

    print(111111, len(data['others']), data['others'])  # , data['others']
    data['others'] = np.array(data['others'])
    max1 = np.array(data['others']).max()
    print(max1)

    # data_folder = "E:/NM-Net-Initial/datasets/COLMAP_no_suffle"
    var_name_list = [
        "xs_4", "label", "img1s", "img2s", "xs_12", "others1"
    ]
    # data_folder = "E:/NM-Net-xiexie/datasets/COLMAP_5a2a95f032a1c655cfe3de62"
    # data_folder = "E:/NM-Net-Initial/datasets/COLMAP_hehe6"
    data_folder = os.path.join(config.datasets_output_path, "COLMAP_hehe6")

    mode_list = ["train", "valid", "test"]
    for var_mode in mode_list:
        data_path = os.path.join(data_folder, var_mode)
        print("路径", data_path)

        for var_name in var_name_list:
            in_file_name = os.path.join(data_path, var_name) + ".pkl"
            with open(in_file_name, "rb") as ifp:
                if var_name in data:
                    data[var_name] += pickle.load(ifp)
                else:
                    data[var_name] = pickle.load(ifp)

    """
            if var_name == "others":
               data_temp = pickle.load(ifp)
                        max2 = np.array(data_temp).max()
                        data_temp += data['others'].max()
                        data[var_name] += data_temp
                        print(5555, data_temp)
                    else:
    """

    data['others1'] = np.array(data['others1'])
    print(222222, len(data['others1']), data['others1'])
    data['others1'] += max1
    max2 = np.array(data['others1']).max()
    print(max2)
    # data['others'] += data['others1']
    data['others'] = np.concatenate((data['others'], data['others1']), axis=0)
    print(222222, len(data['others1']), data['others'].shape, data['others1'])

    var_name_list = [
        "xs_4", "label", "img1s", "img2s", "xs_12", "others2"
    ]
    # data_folder = "E:/NM-net-xiexie/datasets/COLMAP_900数据集成功方案"
    # data_folder = "E:/NM-Net-Initial/datasets/COLMAP_hehe8"
    data_folder = os.path.join(config.datasets_output_path, "COLMAP_hehe8")

    mode_list = ["train", "valid", "test"]
    for var_mode in mode_list:
        data_path = os.path.join(data_folder, var_mode)
        print("路径", data_path)

        for var_name in var_name_list:
            in_file_name = os.path.join(data_path, var_name) + ".pkl"
            with open(in_file_name, "rb") as ifp:
                if var_name in data:
                    data[var_name] += pickle.load(ifp)
                else:
                    data[var_name] = pickle.load(ifp)

    data['others2'] = np.array(data['others2'])
    print(333333, len(data['others2']), data['others2'])
    data['others2'] += max2
    # data['others'] += data['others2']
    data['others'] = np.concatenate((data['others'], data['others2']), axis=0)
    print(333333, len(data['others2']), data['others'].shape, data['others2'])

    print("111111", len(data["xs_4"]), len(data['others']))
    xs_4 = []
    label = []
    img1s = []
    img2s = []
    xs_12 = []
    others = []

    # len(data["xs_4"])):
    for i in trange(len(data["xs_4"])):
        xs_4 += [data["xs_4"][i]]
        label += [data["label"][i]]
        img1s += [data["img1s"][i]]
        img2s += [data["img2s"][i]]
        xs_12 += [data["xs_12"][i]]
        others += [data["others"][i]]

    shuffle_list = list(zip(xs_4, label, img1s, img2s, xs_12, others))
    random.shuffle(shuffle_list)

    xs_4, label, img1s, img2s, xs_12, others = zip(*shuffle_list)

    var_name_list = ["xs_4", "label", "img1s", "img2s", "xs_12", "others"]
    data = {}
    data["xs_4"] = xs_4[:int(0.7 * len(xs_4))]
    data["label"] = label[:int(0.7 * len(xs_4))]
    data["img1s"] = img1s[:int(0.7 * len(xs_4))]
    data["img2s"] = img2s[:int(0.7 * len(xs_4))]
    data["xs_12"] = xs_12[:int(0.7 * len(xs_4))]
    data["others"] = others[:int(0.7 * len(xs_4))]

    print('Size of training data', len(data["xs_4"]))

    # data_folder = "E:/NM-Net-xiexie/datasets/COLMAP"
    data_folder = os.path.join(config.datasets_output_path, "COLMAP_5")

    train_data_path = os.path.join(data_folder, "train")
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)

    if 1:
        for var_name in var_name_list:
            in_file_name = os.path.join(train_data_path, var_name) + ".pkl"
            with open(in_file_name, "wb") as ofp:
                if 0:
                    pickle.dump(data[var_name], ofp)
                else:
                    joblib.dump(data[var_name], ofp)

    data = {}
    data["xs_4"] = xs_4[int(0.7 * len(xs_4)): int(0.85 * len(xs_4))]
    data["label"] = label[int(0.7 * len(xs_4)): int(0.85 * len(xs_4))]
    data["img1s"] = img1s[int(0.7 * len(xs_4)): int(0.85 * len(xs_4))]
    data["img2s"] = img2s[int(0.7 * len(xs_4)): int(0.85 * len(xs_4))]
    data["xs_12"] = xs_12[int(0.7 * len(xs_4)): int(0.85 * len(xs_4))]
    data["others"] = others[int(0.7 * len(xs_4)): int(0.85 * len(xs_4))]

    print('Size of validation data', len(data["xs_4"]))

    valid_data_path = os.path.join(data_folder, "valid")
    if not os.path.exists(valid_data_path):
        os.makedirs(valid_data_path)

    if 1:
        for var_name in var_name_list:
            in_file_name = os.path.join(valid_data_path, var_name) + ".pkl"
            with open(in_file_name, "wb") as ofp:
                if 0:
                    pickle.dump(data[var_name], ofp)
                else:
                    joblib.dump(data[var_name], ofp)

    data = {}
    data["xs_4"] = xs_4[int(0.85 * len(xs_4)): len(xs_4)]
    data["label"] = label[int(0.85 * len(xs_4)): len(xs_4)]
    data["img1s"] = img1s[int(0.85 * len(xs_4)): len(xs_4)]
    data["img2s"] = img2s[int(0.85 * len(xs_4)): len(xs_4)]
    data["xs_12"] = xs_12[int(0.85 * len(xs_4)): len(xs_4)]
    data["others"] = others[int(0.85 * len(xs_4)): len(xs_4)]

    print('Size of testing data', len(data["xs_4"]))

    test_data_path = os.path.join(data_folder, "test")
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)

    if 1:
        for var_name in var_name_list:
            in_file_name = os.path.join(test_data_path, var_name) + ".pkl"
            with open(in_file_name, "wb") as ofp:
                if 0:
                    pickle.dump(data[var_name], ofp)
                else:
                    joblib.dump(data[var_name], ofp)


def load_data_merge(config, data_name, var_mode):
    print("Loading {} data".format(var_mode))
    data_folder = config.data_dump_prefix
    data_folder = "E:/NM-net-xiexie/datasets"
    var_mode = "test"
    data_path = os.path.join(data_folder, data_name, var_mode)
    print("111111", data_path)
    var_name_list = ["img1s", "img2s", "others"]
    data = {}

    for var_name in var_name_list:
        in_file_name = os.path.join(data_path, var_name) + ".pkl"
        with open(in_file_name, "rb") as ifp:
            if var_name in data:
                if 0:
                    data[var_name] += pickle.load(ifp)
                else:
                    data[var_name] += joblib.load(ifp)
            else:
                if 0:
                    data[var_name] = pickle.load(ifp)
                else:
                    data[var_name] = joblib.load(ifp)
                    print(var_name, len(data[var_name]))

    print("merge>>>>>>>>", len(data["img1s"]))
    merge_data = {}
    for index in trange(len(data["img1s"])):
        # print(data["others"][index])
        left_image_index = data["others"][index][0]
        right_image_index = data["others"][index][1]
        # print(left_image_index, right_image_index, type(left_image_index))

        if left_image_index in data:
            pass
        else:
            merge_data[left_image_index] = data["img1s"][index]

        if right_image_index in data:
            pass
        else:
            merge_data[right_image_index] = data["img2s"][index]

    # merge_data_path = "E:/NM-net-xiexie/datasets/COLMAP/valid/merge_data.pkl"
    merge_data_path = "E:/NM-net-xiexie/datasets/Hpatch_Data/test/merge_data.pkl"

    joblib.dump(merge_data, merge_data_path)

    return data

