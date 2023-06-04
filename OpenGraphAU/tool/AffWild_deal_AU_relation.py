import numpy as np
import os
from dataset_process import *

name_dataset = 'affwild'
list_path_prefix, img_path_vita, label_root, train_path, val_path, test_path = set_your_paths(name_dataset)

class_num = 12

for i in range(1,4):
    read_list_name = 'AffWild2_train_label'+'.txt'
    save_list_name = 'AffWild2_train_AU_relation'+'.txt'
    aus = np.loadtxt(os.path.join(list_path_prefix,read_list_name))
    le = aus.shape[0]
    new_aus = np.zeros((le, class_num * class_num))
    for j in range(class_num):
        for k in range(class_num):
            new_aus[:,j*class_num+k] = 2 * aus[:,j] + aus[:,k]
    np.savetxt(os.path.join(list_path_prefix,save_list_name),new_aus,fmt='%d')