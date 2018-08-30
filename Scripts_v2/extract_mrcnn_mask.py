import os
import argparse
import skimage.io
import numpy as np
from six.moves import cPickle as pickle
import mask_rcnn


def main():
    """
        Extract instance car segmentation results using Mask RCNN model
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        default='../Image/StanfordCars/cars_train'
    )
    parser.add_argument(
        '--segment_dir',
        default='./Segment_MaskRCNN/StanfordCars/cars_train'
    )
    parser.add_argument(
        '--model_path',
        default='./maskrcnn_model/mask_rcnn_coco.h5'
    )
    parser.add_argument(
        '--visualize',
        action='store_true'
    )
    args = parser.parse_args()

    model = mask_rcnn.load_model(args.model_path)

    image_names = os.listdir(args.image_dir)
    image_names = sorted(image_names)
    for image_name in image_names:
        image_file = os.path.join(args.image_dir, image_name)
        image = skimage.io.imread(image_file)
        if len(image.shape) == 2:
            image = np.dstack((image, image, image))
        results = model.detect([image], verbose=1)
        r = results[0]
        if args.visualize:
            mask_rcnn.vis_segmentation(image, r)

        final_result = {'class_ids': [], 'masks': [], 'rois': [], 'scores': []}
        for i in range(len(r['class_ids'])):
            if r['class_ids'][i] == 3 or r['class_ids'][i] == 8:
                final_result['class_ids'].append(r['class_ids'][i])
                final_result['masks'].append(r['masks'][:, :, i])
                final_result['rois'].append(r['rois'][i, :])
                final_result['scores'].append(r['scores'][i])
        if len(final_result['class_ids']) > 0:
            final_result['class_ids'] = np.array(final_result['class_ids'])
            final_result['masks'] = np.array(final_result['masks'])
            final_result['masks'] = np.transpose(final_result['masks'], (1, 2, 0))
            final_result['rois'] = np.array(final_result['rois'])
            final_result['scores'] = np.array(final_result['scores'])
        if args.visualize:
            mask_rcnn.vis_segmentation(image, final_result)

        image_id, ext = os.path.splitext(image_name)
        segment_file = os.path.join(args.segment_dir, image_id + '.pkl')
        with open(segment_file, 'wb') as handle:
            pickle.dump(final_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('extract mask rcnn segment from %s to %s' % (image_file, segment_file))
    print('Done')


if __name__ == '__main__':
    main()
