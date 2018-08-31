import os
import numpy as np
import argparse
from PIL import Image
import skimage.io
import scipy.misc
import time
import deeplab


def main():
    """
        Extract DeepLab semantic segmentation
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        default='../Segmentation/ObjectNet3D/image/car'
    )
    parser.add_argument(
        '--deeplab_model_path',
        default='./deeplab_model/deeplabv3_pascal_trainval_2018_01_04.tar.gz'
    )
    parser.add_argument(
        '--class_label',
        default='car'
    )
    parser.add_argument(
        '--segment_dir',
        default='../Segmentation/ObjectNet3D/gt_mask/car'
    )
    parser.add_argument(
        '--visualize',
        action='store_true'
    )
    args = parser.parse_args()

    if not os.path.exists(args.segment_dir):
        os.makedirs(args.segment_dir)

    deeplab_model = deeplab.DeepLabModel(args.deeplab_model_path)

    files = os.listdir(args.image_dir)
    for file_name in files:
        if not file_name.endswith('jpg') and not file_name.endswith('JPEG'):
            continue
        image_name = os.path.join(args.image_dir, file_name)
        image = Image.open(image_name)
        resized_image, seg_map = deeplab_model.run(image)
        if args.visualize:
            deeplab.vis_segmentation(resized_image, seg_map)
        deeplab_mask = np.zeros_like(seg_map)
        if args.class_label == 'car':
            deeplab_mask[seg_map == 7] = 1
        elif args.class_label == 'aeroplane':
            deeplab_mask[seg_map == 1] = 1
        else:
            raise ValueError('class label must be either car or aeroplane.')
        image_id, ext = os.path.splitext(file_name)
        seg_name = image_id + '.png'
        seg_file_name = os.path.join(args.segment_dir, seg_name)
        scipy.misc.imsave(seg_file_name, deeplab_mask)


if __name__ == '__main__':
    main()
