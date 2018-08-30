import os
import argparse
from PIL import Image
import numpy as np
import scipy.misc
import deeplab


def main():
    """
        Extract semantic car segmentation results using DeepLab v3+ Pascal VOC model
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        default='../Image/StanfordCars/cars_train'
    )
    parser.add_argument(
        '--segment_dir',
        default='./Segment_DeepLab/StanfordCars/cars_train'
    )
    parser.add_argument(
        '--model_path',
        default='./deeplab_model/deeplabv3_pascal_trainval_2018_01_04.tar.gz'
    )
    parser.add_argument(
        '--class_label',
        default='car'
    )
    parser.add_argument(
        '--visualize',
        action='store_true'
    )
    args = parser.parse_args()

    model = deeplab.DeepLabModel(args.model_path)

    image_names = os.listdir(args.image_dir)
    image_names = sorted(image_names)
    for image_name in image_names:
        image_file = os.path.join(args.image_dir, image_name)
        image = Image.open(image_file)
        resized_image, seg_map = model.run(image)
        if args.visualize:
            deeplab.vis_segmentation(resized_image, seg_map)
        final_seg_map = np.zeros_like(seg_map)
        if args.class_label == 'car':
            final_seg_map[seg_map == 7] = 1
        elif args.class_label == 'aeroplane':
            final_seg_map[seg_map == 1] = 1
        else:
            raise ValueError('class label must be either car or aeroplane.')
        image_id, ext = os.path.splitext(image_name)
        segment_file = os.path.join(args.segment_dir, image_id + '.png')
        scipy.misc.imsave(segment_file, final_seg_map)
        print('extract deeplab segment from %s to %s' % (image_file, segment_file))


if __name__ == '__main__':
    main()
