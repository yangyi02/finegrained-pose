'''
    Show the usage of the 3D pose annotation by visualizing a sample annotation.
    The visualization is done by laying the projected 3D model
    onto the 2D image. By default, this script visualize the first training
    image of StanfordCars 3D dataset.
'''


import os
import argparse
import pickle as pkl
import numpy as np
import matplotlib
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

import utils


'''
    Prepare parameter for perspective projection from annotations
    1. compute rotation matrix from angles
    2. add offset to principal point
'''
def gen_proj_param(anno_dict, img_filepath, w_default=300):
    proj_param = {}

    # generate rotation matrix
    azimuth = anno_dict['azimuth']
    elevation = anno_dict['elevation']
    theta = anno_dict['theta']
    rotmat = utils.angle_to_rotmat(azimuth, elevation, theta)
    proj_param['R'] = rotmat

    # add (w/2, h/2) offset to principal point (u, v)
    # note that the image is assumed to resize to width w_default=300 pixel
    with Image.open(img_filepath) as img:
        w_org = img.width
        h_org = img.height
    ratio = w_default / float(w_org)
    h = int(h_org * ratio)
    proj_param['uv'] = np.array([
        anno_dict['u'] + w_default/2,
        anno_dict['v'] + h/2
    ])

    # other parameters as they are
    proj_param['d'] = anno_dict['distance']
    proj_param['f'] = anno_dict['f']
    return proj_param


'''
    Visualize a single sample
'''
def visualize(param, w_default=300):
    image_file = param['image_file']
    model_file = param['model_file']
    proj_param = param['proj_param']

    # load image
    if not os.path.exists(os.path.join(image_file)):
        print "Image file does not exsists. Skip %s"%(image_file)
        return
    img = Image.open(image_file)
    print "Processing %s"%(image_file)
    w = w_default
    h = int(w * img.height / float(img.width))
    img = img.resize((w, h), Image.ANTIALIAS)
    img_array = np.array(img)

    # load CAD model
    if not os.path.exists(os.path.join(model_file)):
        print "Model file does not exsists. Skip %s"%(model_file)
        return
    data_3d = utils.read_obj(model_file)
    vertices_3d = data_3d['vertices']
    faces = data_3d['faces']

    # projection
    vertices_2d = utils.proj(
        vertices_3d, proj_param['R'], proj_param['d'],
        proj_param['uv'], proj_param['f'])

    # lay projection onto the image
    num_vertices = vertices_2d.shape[0]
    num_faces = faces.shape[0]
    x_min = np.min(vertices_2d[:, 0])
    y_min = np.min(vertices_2d[:, 1])
    x_max = np.max(vertices_2d[:, 0])
    y_max = np.max(vertices_2d[:, 1])
    patches = []
    for i in range(0, num_faces):
        trivert = np.zeros((3, 2))
        trivert[0, :] = vertices_2d[faces[i, 0], :]
        trivert[1, :] = vertices_2d[faces[i, 1], :]
        trivert[2, :] = vertices_2d[faces[i, 2], :]
        triangle = Polygon(trivert)
        patches.append(triangle)
    p = PatchCollection(patches, alpha=0.2, linewidth=0)
    fig,ax = plt.subplots()
    plt.imshow(img_array)
    ax.add_collection(p)
    plt.axis('off')
    plt.show()
    plt.close(fig)


'''
    Visualize the first sample of a set of annotations
'''
def main():
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
    args = parser.parse_args()

    # load annotation
    with open(args.anno_file, 'rb') as f:
        annos = pkl.load(f)

    # prepare the parameters to visualize the first sample
    key = sorted(annos.keys())[0]
    anno = annos[key]
    param = {}
    param['image_file'] = os.path.join(args.image_dir, key)
    param['model_file'] = os.path.join(
        args.model_dir, anno['model_id'], 'model.obj')
    param['proj_param'] = gen_proj_param(anno, param['image_file'])

    # visualize
    visualize(param)


if __name__ == '__main__':
    main()
