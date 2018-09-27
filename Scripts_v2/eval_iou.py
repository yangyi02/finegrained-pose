import os
import numpy as np
import argparse
from PIL import Image
import scipy.misc


def compute_iou(segment1, segment2):
    assert(len(segment1.shape) == 2 and len(segment2.shape) == 2)
    if not segment1.shape[0] == segment2.shape[0] or not segment1.shape[1] == segment2.shape[1]:
        segment2 = scipy.misc.imresize(segment2, segment1.shape, 'nearest')
    if np.max(segment1) == 255:
        segment1 = segment1 / 255
    segment1 = segment1.astype(np.bool)
    if np.max(segment2) == 255:
        segment2 = segment2 / 255
    segment2 = segment2.astype(np.bool)
    inter = np.logical_and(segment1, segment2)
    union = np.logical_or(segment1, segment2)
    iou = np.sum(inter) * 1.0 / np.sum(union)
    return iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--segment_dir',
        default='../Segmentation/FGVC_Aircraft/mask'
    )
    parser.add_argument(
        '--gt_segment_dir',
        default='../Segmentation/FGVC_Aircraft/gt_mask'
    )
    args = parser.parse_args()

    files = os.listdir(args.gt_segment_dir)

    iou_all = []
    for file_name in files:
        gt_seg_name = os.path.join(args.gt_segment_dir, file_name)
        gt_seg = np.array(Image.open(gt_seg_name))
        if len(gt_seg.shape) > 2:
            gt_seg = gt_seg[:, :, 1]
        seg_name = os.path.join(args.segment_dir, file_name)
        seg = np.array(Image.open(seg_name))
        if len(seg.shape) > 2:
            seg = seg[:, :, 1]
        iou = compute_iou(seg, gt_seg)
        iou_all.append(iou)
    iou_all = np.array(iou_all)
    iou_ave = np.mean(iou_all)
    iou_std = np.std(iou_all)
    print(iou_all)
    print(iou_ave, iou_std)


if __name__ == '__main__':
    main()
