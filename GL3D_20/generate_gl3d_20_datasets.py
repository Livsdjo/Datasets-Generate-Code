"""
     作为数据集生成汇总代码

     每一个场景的数据集的生成代码是不一样的
     每一个场景的数据集生成代码用一个python文件所写
     以后就以这个汇总代码为总
"""
import merge
from config import get_config, print_usage
config, unparsed = get_config()

import generate_dataset1
import generate_dataset2
import generate_dataset3

if __name__ == '__main__':
    config, unparsed = get_config()
    generate_dataset1.dataset_generate_main(config)
    generate_dataset2.dataset_generate_main(config)
    generate_dataset3.dataset_generate_main(config)
    merge.data_merge(config)
    print("Finish!!!!")
 