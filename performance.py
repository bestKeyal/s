import os
import time
import math
import pathlib
from functools import reduce
from collections import Counter

import utils
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
"""
Traceback (most recent call last):
  File "/kaggle/working/s/train_segment.py", line 34, in <module>
    Seg.train()
  File "/kaggle/working/s/segment.py", line 294, in train
    m_dice = self.invalid(epoch)
  File "/kaggle/working/s/segment.py", line 325, in invalid
    m_dice, m_iou, m_precision, m_recall = performance.save_performance_to_csv(
  File "/kaggle/working/s/performance.py", line 100, in save_performance_to_csv
    TP, TN, FP, FN, accuracy, precision, recall, IOU, DICE, VOE, RVD, specificity = calc_performance(
  File "/kaggle/working/s/performance.py", line 70, in calc_performance
    pred_image = prepro_image(pred_path, img_resize, threshold)
  File "/kaggle/working/s/performance.py", line 22, in prepro_image
    bin_image = np.array(bin_image / 255, dtype=np.int)
  File "/opt/conda/lib/python3.10/site-packages/numpy/__init__.py", line 324, in __getattr__
    raise AttributeError(__former_attrs__[attr])
AttributeError: module 'numpy' has no attribute 'int'.
`np.int` was a deprecated alias for the builtin `int`. To avoid this error in existing code, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations. Did you mean: 'inf'?
"""

"""
Traceback (most recent call last):
  File "/kaggle/working/s/train_segment.py", line 34, in <module>
    Seg.train()
  File "/kaggle/working/s/segment.py", line 294, in train
    m_dice = self.invalid(epoch)
  File "/kaggle/working/s/segment.py", line 325, in invalid
    m_dice, m_iou, m_precision, m_recall = performance.save_performance_to_csv(
  File "/kaggle/working/s/performance.py", line 121, in save_performance_to_csv
    pred_paths[file_index], gt_paths[file_index], img_resize, threshold)
IndexError: list index out of range
"""

"""
Traceback (most recent call last):
  File "/kaggle/working/s/train_segment.py", line 34, in <module>
    Seg.train()
  File "/kaggle/working/s/segment.py", line 294, in train
    m_dice = self.invalid(epoch)
  File "/kaggle/working/s/segment.py", line 325, in invalid
    m_dice, m_iou, m_precision, m_recall = performance.save_performance_to_csv(
  File "/kaggle/working/s/performance.py", line 160, in save_performance_to_csv
    analysis_pd = analysis_pd.append({
  File "/opt/conda/lib/python3.10/site-packages/pandas/core/generic.py", line 6296, in __getattr__
    return object.__getattribute__(self, name)
AttributeError: 'DataFrame' object has no attribute 'append'. Did you mean: '_append'?
add Codeadd Markdown
"""

"""
Traceback (most recent call last):
  File "/kaggle/working/s/train_segment.py", line 34, in <module>
    Seg.train()
  File "/kaggle/working/s/segment.py", line 294, in train
    m_dice = self.invalid(epoch)
  File "/kaggle/working/s/segment.py", line 325, in invalid
    m_dice, m_iou, m_precision, m_recall = performance.save_performance_to_csv(
  File "/kaggle/working/s/performance.py", line 151, in save_performance_to_csv
    TP, TN, FP, FN, accuracy, precision, recall, IOU, DICE, VOE, RVD, specificity = calc_performance(
  File "/kaggle/working/s/performance.py", line 119, in calc_performance
    pred_image = prepro_image(pred_path, img_resize, threshold)
  File "/kaggle/working/s/performance.py", line 67, in prepro_image
    if len(image.shape) != 2:
AttributeError: 'NoneType' object has no attribute 'shape'
"""

def prepro_image(img_path, img_resize, threshold=128):

    print("prepro image\n img_path: ", img_path)
    image = cv.imread(img_path, 0)
    if len(image.shape) != 2:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, img_resize)
    _, bin_image = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
    bin_image = np.array(bin_image / 255, dtype=int)
    return bin_image


def true_positive(pred, gt):
    assert pred.shape == gt.shape
    tp_bool = np.logical_and(pred, gt)
    tp_int = np.array(tp_bool, dtype=int)
    tp = np.sum(tp_int)
    return tp


def true_negative(pred, gt):
    assert pred.shape == gt.shape
    no_pred = np.array(np.logical_not(pred), dtype=int)
    no_gt = np.array(np.logical_not(gt), dtype=int)
    tn = true_positive(no_pred, no_gt)
    return tn


def false_positive(pred, gt):
    assert pred.shape == gt.shape
    no_gt = np.array(np.logical_not(gt), dtype=int)
    fp = true_positive(pred, no_gt)
    return fp


def false_negative(pred, gt):
    assert pred.shape == gt.shape
    no_pred = np.array(np.logical_not(pred), dtype=int)
    fn = true_positive(no_pred, gt)
    return fn


def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)


def calc_performance(pred_path, gt_path, img_resize, threshold=128):
    """ Used to calculate the difference between the segmented image and Ground Truth for statistical prediction
        to evaluate the performance of the model, but currently only gray-scale images can be calculated
     :param pred_path: path of predicted image
     :param gt_path: real mask path
     :param img_resize: the size to resize the image
     :param threshold: Used to binaries images
     :return: pix-accuracy, precision, recall, VOE, RVD, Dice, IOU evaluation indicators
     """
    total_pix = reduce(lambda x, y: x * y, img_resize)
    pred_image = prepro_image(pred_path, img_resize, threshold)
    mask_image = prepro_image(gt_path, img_resize, threshold)

    tp = true_positive(pred_image, mask_image)
    tn = true_negative(pred_image, mask_image)
    fp = false_positive(pred_image, mask_image)
    fn = false_negative(pred_image, mask_image)

    accuracy = (tp + tn) / total_pix
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    iou = tp / (tp + fp + fn + 1e-10)
    dice = 2 * tp / (fn + tp + tp + fp + 1e-10)
    voe = 1 - tp / (tp + fn + fp + 1e-10)
    rvd = (fp - fn) / (fn + tp + 1e-10)
    specificity = tn / (tn + fp + 1e-10)

    return tp, tn, fp, fn, accuracy, precision, recall, iou, dice, voe, rvd, specificity


def save_performance_to_csv(pred_dir, gt_dir, img_resize, csv_save_name, csv_save_path='', threshold=128):
    gt_paths, gt_names = utils.get_path(gt_dir, need_img=True)
    pred_paths, pred_names = utils.get_path(pred_dir, need_img=True)

    record_pd = pd.DataFrame(columns=[
        'pred_name', 'gt_name', 'TP', 'FP', 'FN', 'TN',
        'accuracy', 'precision', 'recall', 'IOU', 'DICE', 'VOE', 'RVD', 'specificity',
    ])

    total_file_nums = len(gt_paths)
    pred_file_nums = len(pred_paths)
    for file_index in tqdm(range(total_file_nums), total=total_file_nums):
        if file_index == pred_file_nums: break
        TP, TN, FP, FN, accuracy, precision, recall, IOU, DICE, VOE, RVD, specificity = calc_performance(
            pred_paths[file_index], gt_paths[file_index], img_resize, threshold)  # TODO 修复这里的BUG

        record_pd = record_pd._append({
            'pred_name': pred_names[file_index],
            'gt_name': gt_names[file_index],
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'IOU': IOU,
            'DICE': DICE,
            'VOE': VOE,
            'RVD': RVD,
            'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN
        }, ignore_index=True)

    record_pd.to_csv(
        os.path.join(csv_save_path, '{}.csv'.format(csv_save_name)), index=True, header=True)

    m_accuracy, m_precision, m_recall, m_iou, m_dice, m_voe, m_rvd, m_spec = analysis_performance(
        os.path.join(csv_save_path, '{}.csv'.format(csv_save_name)))
    analysis_pd = pd.DataFrame(columns=[
        'm_accu', 'm_prec', 'm_recall', 'm_iou', 'm_dice', 'm_voe', 'm_rvd', 'm_spec'
    ])
    analysis_pd = analysis_pd._append({
        'm_accu': m_accuracy, 'm_prec': m_precision, 'm_recall': m_recall, 'm_iou': m_iou,
        'm_dice': m_dice, 'm_voe': m_voe, 'm_rvd': m_rvd, 'm_spec': m_spec,
    }, ignore_index=True)
    analysis_pd.to_csv(
        os.path.join(csv_save_path, 'analysis_{}.csv'.format(csv_save_name)), index=True, header=True)
    return m_dice, m_iou, m_precision, m_recall


def analysis_performance(csv_file_path):
    data_frame = pd.read_csv(csv_file_path, header=None)

    m_accuracy = np.mean(np.array(data_frame.loc[1:, 7], dtype=np.float32))
    m_precision = np.mean(np.array(data_frame.loc[1:, 8], dtype=np.float32))
    m_recall = np.mean(np.array(data_frame.loc[1:, 9], dtype=np.float32))
    m_iou = np.mean(np.array(data_frame.loc[1:, 10], dtype=np.float32))
    m_dice = np.mean(np.array(data_frame.loc[1:, 11], dtype=np.float32))
    m_voe = np.mean(np.array(data_frame.loc[1:, 12], dtype=np.float32))
    m_rvd = np.mean(np.array(data_frame.loc[1:, 13], dtype=np.float32))
    m_spec = np.mean(np.array(data_frame.loc[1:, 14], dtype=np.float32))
    print(
        ' accuracy: {},\n precision: {},\n recall: {},\n iou: {},\n dice: {},\n voe: {},\n rvd: {},\n spec: {}.\n'.format(
            m_accuracy, m_precision, m_recall, m_iou, m_dice, m_voe, m_rvd, m_spec))
    return m_accuracy, m_precision, m_recall, m_iou, m_dice, m_voe, m_rvd, m_spec
