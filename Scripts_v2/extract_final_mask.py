import os
import numpy as np
import argparse
import skimage.io
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import scipy.misc
import sample_visualize
from utils import compute_iou


def main():
    """
        Extract instance car segmentation results by merging DeepLab and Mask RCNN results
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        default='../Image/StanfordCars/cars_train'
    )
    parser.add_argument(
        '--deeplab_segment_dir',
        default='./Segment_DeepLab/StanfordCars/cars_train'
    )
    parser.add_argument(
        '--mrcnn_segment_dir',
        default='./Segment_MaskRCNN/StanfordCars/cars_train'
    )
    parser.add_argument(
        '--segment_dir',
        default='./Segment_Final/StanfordCars/cars_train'
    )
    parser.add_argument(
        '--anno_file',
        default='../Anno3D/StanfordCars/train_anno.pkl'
    )
    parser.add_argument(
        '--model_dir',
        default='../CAD/02958343'
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

    # load annotation
    with open(args.anno_file, 'rb') as f:
        annos = pickle.load(f)

    for key in sorted(annos.keys()):
        anno = annos[key]
        param = {}
        param['image_file'] = os.path.join(args.image_dir, key)
        param['model_file'] = os.path.join(
            args.model_dir, anno['model_id'], 'model.obj')
        param['proj_param'] = sample_visualize.gen_proj_param(anno, param['image_file'])
        image, mask = sample_visualize.get_binary_mask(param)

        image_id, ext = os.path.splitext(key)

        deeplab_file = os.path.join(args.deeplab_segment_dir, image_id + '.png')
        deeplab_segment = skimage.io.imread(deeplab_file)
        deeplab_segment = scipy.misc.imresize(deeplab_segment, mask.shape, 'nearest')

        mrcnn_file = os.path.join(args.mrcnn_segment_dir, image_id + '.pkl')
        with open(mrcnn_file, 'rb') as handle:
            mrcnn_segment = pickle.load(handle)

        # Search for best overlapped instance segmentation mask
        best_iou = 0
        for i in range(len(mrcnn_segment['class_ids'])):
            if args.class_label == 'car':
                assert(mrcnn_segment['class_ids'][i] == 3 or mrcnn_segment['class_ids'][i] == 8)
            elif args.class_label == 'aeroplane':
                assert(mrcnn_segment['class_ids'][i] == 5)
            else:
                raise ValueError('class label must be either car or aeroplane.')
            iou = compute_iou(mrcnn_segment['masks'][:, :, i], mask)
            if iou > best_iou:
                final_segment = mrcnn_segment['masks'][:, :, i].astype(np.uint8)
                final_segment = scipy.misc.imresize(final_segment, mask.shape, 'nearest')
                best_iou = iou

        # If there is only one object in the image, also compare with deeplab results
        if len(mrcnn_segment['class_ids']) <= 1:
            iou = compute_iou(deeplab_segment, mask)
            if iou > best_iou:
                final_segment = deeplab_segment
                best_iou = iou
                print('Use deeplab results %s' % deeplab_file)

        final_segment = final_segment.astype(np.uint8)
        if args.visualize:
            plt.subplots()
            plt.imshow(final_segment)
            plt.show()

        segment_file = os.path.join(args.segment_dir, image_id + '.png')
        if np.max(final_segment) < 1.01:
            final_segment = final_segment * 255
        segment = np.dstack((final_segment, final_segment, final_segment))
        scipy.misc.imsave(segment_file, segment)
        print('merge segment from %s and %s to %s' % (deeplab_file, mrcnn_file, segment_file))
        # break


if __name__ == '__main__':
    main()
