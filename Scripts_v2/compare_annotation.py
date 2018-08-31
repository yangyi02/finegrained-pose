"""
    Show the usage of the 3D pose annotation by visualizing a sample annotation.
    The visualization is done by laying the projected 3D model
    onto the 2D image. By default, this script visualize the first training
    image of StanfordCars 3D dataset.
"""
import os
import sys
import argparse
import pickle as pkl
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import utils
from OpenGL import GL
from OpenGL.GL import *
import cyglfw3 as glfw
import glutils
import scipy.misc


def gen_proj_param(anno_dict, img_file_path, w_default=300):
    """
        Prepare parameter for perspective projection from annotations
        1. compute rotation matrix from angles
        2. add offset to principal point
    """
    proj_param = {}

    # generate rotation matrix
    azimuth = anno_dict['azimuth']
    elevation = anno_dict['elevation']
    theta = anno_dict['theta']
    rotmat = utils.angle_to_rotmat(azimuth, elevation, theta)
    proj_param['R'] = rotmat

    # add (w/2, h/2) offset to principal point (u, v)
    # note that the image is assumed to resize to width w_default=300 pixel
    with Image.open(img_file_path) as img:
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


def get_faces_vertices_2d(param):
    model_file = param['model_file']
    proj_param = param['proj_param']

    # load CAD model
    if not os.path.exists(os.path.join(model_file)):
        print("Model file does not exsists. Skip %s" % model_file)
        return
    data_3d = utils.read_obj(model_file)
    vertices_3d = data_3d['vertices']
    faces = data_3d['faces']

    # projection
    vertices_2d = utils.proj(vertices_3d, proj_param['R'], proj_param['d'],
                             proj_param['uv'], proj_param['f'])
    return faces, vertices_2d


def visualize_polygon(param, w_default=300):
    """
        Visualize a single sample
    """
    faces, vertices_2d = get_faces_vertices_2d(param)

    image_file = param['image_file']
    # load image
    if not os.path.exists(os.path.join(image_file)):
        print("Image file does not exsists. Skip %s" % image_file)
        return
    img = Image.open(image_file)
    print("Processing %s" % image_file)
    w = w_default
    h = int(w * img.height / float(img.width))
    img = img.resize((w, h), Image.ANTIALIAS)
    img_array = np.array(img)

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
    fig, ax = plt.subplots()
    plt.imshow(img_array)
    ax.add_collection(p)
    plt.axis('off')
    plt.show()


def generate_binary_mask(faces, vertices_2d, width, height):
    if not glfw.Init():
        print('glfw not initialized')
        sys.exit()

    version = 3, 3
    glfw.WindowHint(glfw.CONTEXT_VERSION_MAJOR, version[0])
    glfw.WindowHint(glfw.CONTEXT_VERSION_MINOR, version[1])
    glfw.WindowHint(glfw.OPENGL_FORWARD_COMPAT, 1)
    glfw.WindowHint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.WindowHint(glfw.VISIBLE, 1)

    window = glfw.CreateWindow(width, height, 'Quad')
    if not window:
        print('glfw window not created')
        glfw.Terminate()
        sys.exit()

    glfw.MakeContextCurrent(window)

    strVS = """
        #version 330
        layout(location = 0) in vec2 aPosition;

        void main() {
            gl_Position = vec4(vec3(aPosition, 0), 1.0);
        }
        """
    strFS = """
        #version 330
        out vec3 fragColor;

        void main() {
            fragColor = vec3(0, 1, 0);
        }
        """

    program = glutils.loadShaders(strVS, strFS)
    glUseProgram(program)

    element_array = np.reshape(faces, -1)
    elementData = np.array(element_array, np.uint32)

    vertex_array = np.reshape(vertices_2d, -1)
    vertexData = np.array(vertex_array, np.float32)

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * len(elementData), elementData, GL_STATIC_DRAW)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, 4 * len(vertexData), vertexData, GL_STATIC_DRAW)

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

    glBindVertexArray(0)

    GL.glClearColor(0, 0, 0, 1.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

    glUseProgram(program)
    glBindVertexArray(vao)
    glDrawElements(GL_TRIANGLES, len(element_array), GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    pixel_data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, outputType=None)

    im = np.array(pixel_data)
    mask = Image.frombuffer('RGB', (width, height), im, 'raw', 'RGB')
    mask = np.array(mask)
    glfw.Terminate()
    return mask


def get_binary_mask(param, w_default=300):
    faces, vertices_2d = get_faces_vertices_2d(param)

    image_file = param['image_file']
    # load image
    if not os.path.exists(os.path.join(image_file)):
        print("Image file does not exsists. Skip %s" % image_file)
        return
    img = Image.open(image_file)
    # print("Processing %s" % image_file)
    w = w_default
    h = int(w * img.height / float(img.width))
    img = img.resize((w, h), Image.ANTIALIAS)
    img_array = np.array(img)

    vertices_2d[:, 0] = vertices_2d[:, 0] / w * 2 - 1
    vertices_2d[:, 1] = vertices_2d[:, 1] / h * 2 - 1
    mask = generate_binary_mask(faces, vertices_2d, w, h)
    mask = mask[:, :, 1]
    return img_array, mask


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
        '--new_anno_dir',
        default='../Anno3D_v2/StanfordCars/train_anno'
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
        '--overlay_dir',
        default='../Overlay/StanfordCars/cars_train'
    )
    parser.add_argument(
        '--new_overlay_dir',
        default='../Overlay_v2/StanfordCars/cars_train'
    )
    args = parser.parse_args()

    if not os.path.exists(args.overlay_dir):
        os.makedirs(args.overlay_dir)
    if not os.path.exists(args.new_overlay_dir):
        os.makedirs(args.new_overlay_dir)

    with open(args.anno_file, 'rb') as f:
        annos = pkl.load(f)

    keys = sorted(annos.keys())
    keys = keys[0:]
    for key in keys:
        print('processing %s' % key)
        # prepare the parameters to visualize the sample
        anno = annos[key]
        param = dict()
        param['image_file'] = os.path.join(args.image_dir, key)
        param['model_file'] = os.path.join(args.model_dir, anno['model_id'], 'model.obj')
        param['proj_param'] = gen_proj_param(anno, param['image_file'])

        image, mask = get_binary_mask(param)
        if len(image.shape) == 2:
            image = np.dstack((image, image, image))
        mask = np.dstack((np.zeros_like(mask), mask, np.zeros_like(mask)))  # Make mask a green mask
        image = image.astype(np.float) / 255.0 * 0.8 + mask.astype(np.float) / 255.0 * 0.2
        file_name = os.path.join(args.overlay_dir, key)
        scipy.misc.imsave(file_name, image)

        # load new annotation
        image_id, ext = os.path.splitext(key)
        anno_file = image_id + '.pkl'
        with open(os.path.join(args.new_anno_dir, anno_file), 'rb') as f:
            new_annos = pkl.load(f)

        new_anno = new_annos[key]
        param = dict()
        param['image_file'] = os.path.join(args.image_dir, key)
        param['model_file'] = os.path.join(args.model_dir, new_anno['model_id'], 'model.obj')
        param['proj_param'] = gen_proj_param(new_anno, param['image_file'])

        image, mask = get_binary_mask(param)
        if len(image.shape) == 2:
            image = np.dstack((image, image, image))
        mask = np.dstack((np.zeros_like(mask), mask, np.zeros_like(mask)))  # Make mask a green mask
        image = image.astype(np.float) / 255.0 * 0.8 + mask.astype(np.float) / 255.0 * 0.2
        file_name = os.path.join(args.new_overlay_dir, key)
        scipy.misc.imsave(file_name, image)


if __name__ == '__main__':
    main()
