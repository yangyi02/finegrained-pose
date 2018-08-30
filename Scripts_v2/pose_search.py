import os
import argparse
import pickle as pkl
from PIL import Image
import numpy as np
import scipy.misc
import sample_visualize
from utils import compute_iou
import time


def search_pose(anno, param, segment):
    if len(segment.shape) > 2:
        segment = segment[:, :, 0]

    param['proj_param'] = sample_visualize.gen_proj_param(anno, param['image_file'])
    # mask = sample_visualize.visualize_binary_mask(param)
    image, mask = sample_visualize.get_binary_mask(param)
    iou = compute_iou(mask, segment)
    # print(iou)

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
            # mask = sample_visualize.visualize_binary_mask(param)
            iou1 = compute_iou(mask, segment)

            anno2 = anno.copy()
            anno2[keywords[k]] = anno2[keywords[k]] - alpha * step_size[k]
            param['proj_param'] = sample_visualize.gen_proj_param(anno2, param['image_file'])
            image, mask = sample_visualize.get_binary_mask(param)
            # mask = sample_visualize.visualize_binary_mask(param)
            iou2 = compute_iou(mask, segment)

            if iou1 >= iou and iou1 >= iou2:
                anno[keywords[k]] = anno[keywords[k]] + alpha * step_size[k]
                iou = iou1
            elif iou2 >= iou and iou2 >= iou1:
                anno[keywords[k]] = anno[keywords[k]] - alpha * step_size[k]
                iou = iou2
            # print(cnt, alpha, k, iou)
        cnt = cnt + 1
        if iou == last_iou:
            break
    # param['proj_param'] = sample_visualize.gen_proj_param(anno, param['image_file'])
    # mask = sample_visualize.visualize_binary_mask(param)
    # iou = compute_iou(mask, segment)
    # print(iou)
    return anno


def main():
    """
        Visualize the first sample of a set of annotations
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--anno_file',
        default='../Anno3D/StanfordCars/train_anno.pkl'
    )
    parser.add_argument(
        '--image_dir',
        default='../Image/StanfordCars/cars_train'
    )
    parser.add_argument(
        '--model_dir',
        default='../CAD/02958343'
    )
    parser.add_argument(
        '--segment_dir',
        default='./Segment_Final/StanfordCars/cars_train'
    )
    parser.add_argument(
        '--new_anno_dir',
        default='../Anno3D/StanfordCars/train_anno_new'
    )
    args = parser.parse_args()
    print(args)

    # load annotation
    with open(args.anno_file, 'rb') as f:
        annos = pkl.load(f)

    keys = sorted(annos.keys())
    # keys = keys[690:]
    for key in keys:
        start_time = time.time()
        anno = annos[key]
        param = {}
        param['image_file'] = os.path.join(args.image_dir, key)
        print("Processing %s" % param['image_file'])
        param['model_file'] = os.path.join(
            args.model_dir, anno['model_id'], 'model.obj')
        image_id, ext = os.path.splitext(key)
        param['segment_file'] = os.path.join(args.segment_dir, image_id + '.png')
        segment = np.array(Image.open(param['segment_file']))

        anno = search_pose(anno, param, segment)
        new_anno = dict()
        new_anno[key] = anno
        file_name = image_id + '.pkl'
        with open(os.path.join(args.new_anno_dir, file_name), 'wb') as handle:
            pkl.dump(new_anno, handle)
        elapsed_time = time.time() - start_time
        print('Spend %s' % time.strftime('%H:%M:%S', time.gmtime(elapsed_time)))
        # break


if __name__ == '__main__':
    main()
