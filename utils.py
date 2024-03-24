import os
import time
import math
import pathlib
from functools import reduce
from collections import Counter

import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
"""
Traceback (most recent call last):
  File "/kaggle/working/DR-UNet-Suited/DR-UNet-Suited/train_segment.py", line 34, in <module>
    Seg.train()
  File "/kaggle/working/DR-UNet-Suited/DR-UNet-Suited/segment.py", line 283, in train
    m_dice = self.invalid(epoch)
  File "/kaggle/working/DR-UNet-Suited/DR-UNet-Suited/segment.py", line 310, in invalid
    utils.crop_image(epoch_pred_save_dir, epoch_cropped_save_dir,
  File "/kaggle/working/DR-UNet-Suited/DR-UNet-Suited/utils.py", line 48, in crop_image
    file_paths_list = get_file_path(read_dir)
NameError: name 'get_file_path' is not defined. Did you mean: 'file_path'?

"""

"""
Traceback (most recent call last):
  File "/kaggle/working/DR-UNet-Suited/DR-UNet-Suited/DR-UNet-Suited/train_segment.py", line 34, in <module>
    Seg.train()
  File "/kaggle/working/DR-UNet-Suited/DR-UNet-Suited/DR-UNet-Suited/segment.py", line 283, in train
    m_dice = self.invalid(epoch)
  File "/kaggle/working/DR-UNet-Suited/DR-UNet-Suited/DR-UNet-Suited/segment.py", line 310, in invalid
    utils.crop_image(epoch_pred_save_dir, epoch_cropped_save_dir,
  File "/kaggle/working/DR-UNet-Suited/DR-UNet-Suited/DR-UNet-Suited/utils.py", line 58, in crop_image
    file_paths_list = get_path(read_dir)
  File "/kaggle/working/DR-UNet-Suited/DR-UNet-Suited/DR-UNet-Suited/utils.py", line 31, in get_path
    path_list = sorted(path_list, key=lambda path_: int(pathlib.Path(path_).stem))
  File "/kaggle/working/DR-UNet-Suited/DR-UNet-Suited/DR-UNet-Suited/utils.py", line 31, in <lambda>
    path_list = sorted(path_list, key=lambda path_: int(pathlib.Path(path_).stem))
ValueError: invalid literal for int() with base 10: 'Segment_train_pred_10'
"""

"""
Traceback (most recent call last):
  File "/kaggle/working/s/train_segment.py", line 34, in <module>
    Seg.train()
  File "/kaggle/working/s/segment.py", line 283, in train
    m_dice = self.invalid(epoch)
  File "/kaggle/working/s/segment.py", line 310, in invalid
    utils.crop_image(epoch_pred_save_dir, epoch_cropped_save_dir,
  File "/kaggle/working/s/utils.py", line 95, in crop_image
    img = cv.imread(file_path)
TypeError: Can't convert object to 'str' for 'filename'

"""
"""
Traceback (most recent call last):
  File "/kaggle/working/s/train_segment.py", line 34, in <module>
    Seg.train()
  File "/kaggle/working/s/segment.py", line 283, in train
    m_dice = self.invalid(epoch)
  File "/kaggle/working/s/segment.py", line 310, in invalid
    utils.crop_image(epoch_pred_save_dir, epoch_cropped_save_dir,
  File "/kaggle/working/s/utils.py", line 112, in crop_image
    cropped = img[row * r_h: (row + 1) * r_h, col * r_w: (col + 1) * r_w, :]
TypeError: 'NoneType' object is not subscriptable

"""

import pathlib
import re


def extract_number_at_end(filename):
    # 正则表达式匹配文件名末尾的数字
    match = re.search(r'(\d+)(?!.*\d)', filename)
    if match:
        return int(match.group(0))  # 匹配的数字部分
    else:
        return float('inf')  # 没有数字，则返回无穷大


def get_path(file_dir):
    path_list = []
    name_list = []
    for path in pathlib.Path(file_dir).iterdir():
        path_list.append(os.path.join(file_dir, str(path)))
        name_list.append(path.name)

    # 辅助函数作为key来对路径列表进行排序
    path_list = sorted(path_list, key=lambda path_: extract_number_at_end(pathlib.Path(path_).stem))
    name_list = sorted(name_list, key=lambda name_: extract_number_at_end(name_))

    return path_list, name_list


def check_file(paths):
    if type(paths) is not list:
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
    return


def list_file(dir_path):
    paths = []
    for path in pathlib.Path(dir_path).iterdir():
        paths.append(str(path))
    return paths


def crop_image(read_dir, save_dir, o_w, o_h, r_w, r_h, split=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Crop large images into small images
    i = 0
    file_paths_list = os.listdir(read_dir)
    for file_name in file_paths_list:
        file_path = os.path.join(read_dir, file_name)

        for row in range(o_h // r_h):
            for col in range(o_w // r_w):
                img = cv.imread(file_path)
                cropped = img[row * r_h: (row + 1) * r_h, col * r_w: (col + 1) * r_w, :]

                if split:
                    save_path = os.path.join(save_dir, str(pathlib.Path(file_path).stem))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    cv.imwrite(os.path.join(save_path, '{}.jpg'.format(i)), cropped)
                else:
                    cv.imwrite(os.path.join(save_dir, '{}.jpg'.format(i)), cropped)
                i += 1
    print('Cropped image is complete!')
    return
