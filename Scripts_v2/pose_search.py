import os
import numpy as np
import argparse
from PIL import Image
import skimage.io
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import scipy.misc
import sample_visualize
import time
import maskrcnn
import deeplab


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


def search_pose(anno, param, segment_reference, visualize):
    if len(segment_reference.shape) > 2:
        segment_reference = segment_reference[:, :, 0]

    param['proj_param'] = sample_visualize.gen_proj_param(anno, param['image_file'])
    if visualize:
        mask = sample_visualize.visualize_binary_mask(param)
    else:
        image, mask = sample_visualize.get_binary_mask(param)
    iou = compute_iou(mask, segment_reference)
    if visualize:
        print('initial iou:', iou)

    alpha = 1
    keywords = ['azimuth', 'elevation', 'theta', 'distance', 'f', 'u', 'v']
    step_size = [np.pi/100, np.pi/100, np.pi/100, 0.01, 10, 1, 1]
    cnt = 0
    while True:
        last_iou = iou
        for k in range(7):
            anno1 = anno.copy()
            anno1[keywords[k]] = anno1[keywords[k]] + alpha * step_size[k]
            param['proj_param'] = sample_visualize.gen_proj_param(anno1, param['image_file'])
            image, mask = sample_visualize.get_binary_mask(param)
            iou1 = compute_iou(mask, segment_reference)

            anno2 = anno.copy()
            anno2[keywords[k]] = anno2[keywords[k]] - alpha * step_size[k]
            param['proj_param'] = sample_visualize.gen_proj_param(anno2, param['image_file'])
            image, mask = sample_visualize.get_binary_mask(param)
            iou2 = compute_iou(mask, segment_reference)

            if iou1 >= iou and iou1 >= iou2:
                anno[keywords[k]] = anno[keywords[k]] + alpha * step_size[k]
                iou = iou1
            elif iou2 >= iou and iou2 >= iou1:
                anno[keywords[k]] = anno[keywords[k]] - alpha * step_size[k]
                iou = iou2
            if visualize:
                print(cnt, alpha, k, iou)
        if visualize:
            param['proj_param'] = sample_visualize.gen_proj_param(anno, param['image_file'])
            mask = sample_visualize.visualize_binary_mask(param)
        cnt = cnt + 1
        if iou == last_iou:
            break
    return anno


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
        '--anno_file',
        default='../Anno3D/StanfordCars/train_anno.pkl'
    )
    parser.add_argument(
        '--model_dir',
        default='../CAD/02958343'
    )
    parser.add_argument(
        '--deeplab_model_path',
        default='./deeplab_model/deeplabv3_pascal_trainval_2018_01_04.tar.gz'
    )
    parser.add_argument(
        '--maskrcnn_model_path',
        default='./maskrcnn_model/mask_rcnn_coco.h5'
    )
    parser.add_argument(
        '--class_label',
        default='car'
    )
    parser.add_argument(
        '--new_anno_dir',
        default='../Anno3D/StanfordCars/train_anno_new'
    )
    parser.add_argument(
        '--visualize',
        action='store_true'
    )
    args = parser.parse_args()
    args.visualize = True

    # load annotation
    with open(args.anno_file, 'rb') as f:
        annos = pickle.load(f)

    deeplab_model = deeplab.DeepLabModel(args.deeplab_model_path)
    maskrcnn_model = maskrcnn.load_model(args.maskrcnn_model_path)

    keys = sorted(annos.keys())
    keys = keys[0:50]
    for key in keys:
        start_time = time.time()
        anno = annos[key]
        param = {}
        param['image_file'] = os.path.join(args.image_dir, key)
        print("Processing %s" % param['image_file'])
        param['model_file'] = os.path.join(args.model_dir, anno['model_id'], 'model.obj')
        param['proj_param'] = sample_visualize.gen_proj_param(anno, param['image_file'])
        # Get 2d mask projected from 3d model
        image, mask = sample_visualize.get_binary_mask(param)
        # Get deeplab semantic segmentation
        image = Image.open(param['image_file'])
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
        # Get mask rcnn instance segmentation
        image = skimage.io.imread(param['image_file'])
        if len(image.shape) == 2:
            image = np.dstack((image, image, image))
        results = maskrcnn_model.detect([image], verbose=1)
        r = results[0]
        if args.visualize:
            maskrcnn.vis_segmentation(image, r)
        maskrcnn_mask = {'class_ids': [], 'masks': [], 'rois': [], 'scores': []}
        for i in range(len(r['class_ids'])):
            if args.class_label == 'car':
                if r['class_ids'][i] == 3 or r['class_ids'][i] == 8:
                    maskrcnn_mask['class_ids'].append(r['class_ids'][i])
                    maskrcnn_mask['masks'].append(r['masks'][:, :, i])
                    maskrcnn_mask['rois'].append(r['rois'][i, :])
                    maskrcnn_mask['scores'].append(r['scores'][i])
            elif args.class_label == 'aeroplane':
                if r['class_ids'][i] == 5:
                    maskrcnn_mask['class_ids'].append(r['class_ids'][i])
                    maskrcnn_mask['masks'].append(r['masks'][:, :, i])
                    maskrcnn_mask['rois'].append(r['rois'][i, :])
                    maskrcnn_mask['scores'].append(r['scores'][i])
            else:
                raise ValueError('class label must be either car or aeroplane.')
        if len(maskrcnn_mask['class_ids']) > 0:
            maskrcnn_mask['class_ids'] = np.array(maskrcnn_mask['class_ids'])
            maskrcnn_mask['masks'] = np.array(maskrcnn_mask['masks'])
            maskrcnn_mask['masks'] = np.transpose(maskrcnn_mask['masks'], (1, 2, 0))
            maskrcnn_mask['rois'] = np.array(maskrcnn_mask['rois'])
            maskrcnn_mask['scores'] = np.array(maskrcnn_mask['scores'])
        if args.visualize:
            maskrcnn.vis_segmentation(image, maskrcnn_mask)
        # Get final segmentation reference
        # If there is only one object in the image, also compare with deeplab results
        if len(maskrcnn_mask['class_ids']) <= 1:
            print('Use deeplab result')
            final_segment = deeplab_mask
        else:
            print('Use mask_rcnn result')
            # Search for best overlapped instance segmentation mask
            best_iou = 0
            for i in range(len(maskrcnn_mask['class_ids'])):
                if args.class_label == 'car':
                    assert(maskrcnn_mask['class_ids'][i] == 3 or maskrcnn_mask['class_ids'][i] == 8)
                elif args.class_label == 'aeroplane':
                    assert(maskrcnn_mask['class_ids'][i] == 5)
                else:
                    raise ValueError('class label must be either car or aeroplane.')
                iou = compute_iou(maskrcnn_mask['masks'][:, :, i], mask)
                if iou > best_iou:
                    final_segment = maskrcnn_mask['masks'][:, :, i].astype(np.uint8)
                    best_iou = iou
        final_segment = final_segment.astype(np.uint8)
        if args.visualize:
            plt.subplots()
            plt.imshow(final_segment)
            plt.show()
        # Local greedy search for pose that best match segmentation reference
        anno = search_pose(anno, param, final_segment, args.visualize)
        new_anno = dict()
        new_anno[key] = anno
        image_id, ext = os.path.splitext(key)
        file_name = image_id + '.pkl'
        with open(os.path.join(args.new_anno_dir, file_name), 'wb') as handle:
            pkl.dump(new_anno, handle)
        elapsed_time = time.time() - start_time
        print('Spend %s' % time.strftime('%H:%M:%S', time.gmtime(elapsed_time)))
        break


if __name__ == '__main__':
    main()
